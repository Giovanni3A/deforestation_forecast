import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import f1_score
import torch.nn as nn
from dataloader import load_data, CustomDataset
from model import init_model, train

import logging
from datetime import datetime

if __name__ == "__main__":

    # setup logger
    logfile = f"FeatureSelection_{datetime.now().strftime('%Y%m%d%H%M%S')}.log"
    handler = logging.FileHandler(logfile)
    handler.setFormatter(logging.Formatter("%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s"))
    logger = logging.getLogger("logger")
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)

    # load all data
    (
        train_data, val_data, test_data,
        patches, frames_idx, 
        county_data,
        counties_time_grid,
        precip_time_grid,
        tpi_array,
        landcover_array,
        scores_time_grid,
        night_time_grid,
        sentinel_time_grid
    ) = load_data()

    def generate_dataloaders(channels):
        trainloader = torch.utils.data.DataLoader(
            CustomDataset(
                train_data, 
                patches, 
                frames_idx, 
                county_data,
                counties_time_grid,
                precip_time_grid,
                tpi_array,
                landcover_array,
                scores_time_grid,
                night_time_grid,
                sentinel_time_grid,
                channels
            ),
            batch_size=64,
            shuffle=True
        )

        valloader = torch.utils.data.DataLoader(
            CustomDataset(
                val_data, 
                patches, 
                frames_idx, 
                county_data,
                counties_time_grid,
                precip_time_grid,
                tpi_array,
                landcover_array,
                scores_time_grid,
                night_time_grid,
                sentinel_time_grid,
                channels
            ),
            batch_size=1000,
            shuffle=True
        )
        return trainloader, valloader

    def evaluate_predictions(model, dataloader, treshold=None, treshold_range=range(25, 40)):
        # softmax function will be needed
        softmax_ = nn.Softmax(dim=1)
        # get prediction values as binary
        y_true = []
        y_pred = []
        for inputs, labels in dataloader:
            y_hat = model(inputs).detach()
            y_true.append(labels[:, 1, :, :].cpu())
            y_pred.append(softmax_(y_hat)[:, 1, :, :].cpu())
        # flatten
        y_true = np.concatenate(y_true).flatten() 
        y_pred = np.concatenate(y_pred).flatten()
        if treshold is None:
            # select probability treshold
            scores = []
            for p_treshold in tqdm(treshold_range, desc="Testing treshold values"):
                pt = p_treshold / 100
                scores.append((
                    pt, 
                    f1_score(
                        y_true, 
                        y_pred > pt
                    )
                ))
            # sort by score and get from best treshold
            scores.sort(key=lambda i: i[1])
            best = scores[-1]
            print("Best treshold / score:", best)
            return best
        else:
            score = f1_score(
                y_true, 
                y_pred > treshold
            )
            print("Score:", score)
            return score

    # number of epochs to train each model
    n_epochs = 15

    # list of all available channels
    all_channels = list(range(25))

    def run_selection_iteration(channels):
        # train model with defined starting channels
        print(f"\nEvaluating starting model:")
        trainloader, valloader = generate_dataloaders(channels)
        model, optimizer = init_model(len(channels))
        train(model, optimizer, n_epochs, trainloader, valloader)
        # basic_train_err = np.mean([i[0].cpu() for i in model.errs[-3:]])
        # basic_val_err = np.mean([i[1].cpu() for i in model.errs[-3:]])
        # basic_error = (basic_train_err, basic_val_err)
        # evaluate errors and estimate optimal treshold
        treshold, basic_train_score = evaluate_predictions(model, trainloader, treshold=None)
        basic_val_score = evaluate_predictions(model, valloader, treshold)
        basic_score = (basic_train_score, basic_val_score)
        logger.info(f"Basic scores: {basic_score}")

        # train model for other channels and save performance
        # c_errors = []
        c_scores = []
        for c in all_channels:
            if not(c in channels):
                print(f"\nStarting evaluation of additional feature ({c}):")
                logger.info(f"Starting evaluation of additional feature ({c})")
                c_trainloader, c_valloader = generate_dataloaders(channels+[c])
                c_model, c_optimizer = init_model(len(channels)+1)
                train(c_model, c_optimizer, n_epochs, c_trainloader, c_valloader)
                # avg_train_err = np.mean([i[0].cpu() for i in c_model.errs[-3:]])
                # avg_val_err = np.mean([i[1].cpu() for i in c_model.errs[-3:]])
                # c_errors.append((c, avg_train_err, avg_val_err))
                c_treshold, c_train_score = evaluate_predictions(c_model, c_trainloader, treshold=None)
                c_val_score = evaluate_predictions(c_model, c_valloader, c_treshold)
                c_scores.append((c, c_train_score, c_val_score))
                logger.info(f"Best treshold: {c_treshold}")
                logger.info(f"Train score: {c_train_score}")
                logger.info(f"Validation score: {c_val_score}")

        # get new channel that decreases (validation) loss the most
        # best_decrease = 0
        best_increase = 0
        best_channel = None
        # best_errors = None
        best_scores = None
        for (c, score_train, score_val) in c_scores:
            # decrease = float(basic_error[1] - err_val)
            increase = float(score_val - basic_val_score)
            if increase > best_increase:
                # best_decrease = decrease
                best_increase = increase
                best_channel = c
                # best_errors = (err_train, err_val)
                best_scores = (score_train, score_val)
            
        return best_channel, basic_score, best_scores

    # start with basic features
    channels = [0, 1, 2, 3]

    # main selection loop
    iter = 1
    errors_hist = []
    while True:
        print(f"\n\n Selection loop iteration {iter} | Channels: {channels}")
        logger.info(f"Selection loop iteration {iter} | Initial channels: {channels}")
        new_channel, starting_scores, final_scores = run_selection_iteration(channels)
        errors_hist.append((starting_scores, final_scores))
        if new_channel is None:
            logger.info("New iteration didn't find any channel that would increase model score")
            logger.info(f"Final channels: {channels}")
            break
        else:
            logger.info(f"New channel added: {new_channel} ({starting_scores} -> {final_scores})")
            channels.append(new_channel)
        iter += 1

    print("\nFinal channels results:", channels)

    # save results to file
    errors_hist_df = []
    for (e1, e2) in errors_hist:
        i = [float(e1[0]), float(e1[1])]
        if e2 is not None:
            i = i + [float(e2[0]), float(e2[1])]
        else:
            i = i + [np.nan, np.nan]
        errors_hist_df.append(i)
    errors_hist_df = pd.DataFrame(
        errors_hist_df, 
        columns=["base_train", "base_val", "improved_train", "improved_val"]
    )
    errors_hist_df.to_csv("scores_history.csv", index=False)