import logging
from time import sleep, time
from itertools import product
from pathlib import Path as P
from tqdm import tqdm
import yaml

import torch
import ultralytics


def train_model(
    model: ultralytics.models.yolo.model.YOLO, 
    training_args: dict, 
    tuning_args: dict, 
    run_num: int
) -> bool:
    merged_args = {**training_args, **tuning_args}

    try:
        t0 = time()
        result: ultralytics.utils.metrics.DetMetrics = model.train(**merged_args)

        # Print results
        logging.info("========== TRAINING RESULT FOR %s==========", training_args["name"])
        logging.info("Time taken: %.2f seconds", time() - t0)

        if result is not None:
            precision = result.results_dict["metrics/precision(B)"]
            recall = result.results_dict["metrics/recall(B)"]
            f1 = 2 * precision * recall / (precision + recall)
            logging.info("Overall F1 score: %.2f", f1)

        logging.info("========== END TRAINING RESULT ==========")
        sleep(1)

    except torch.cuda.OutOfMemoryError as e:
        logging.error("CUDA out of memory error in training run %d", run_num)
        logging.error(e)
        torch.cuda.empty_cache()
        return False

    except KeyboardInterrupt as e:
        logging.error("KeyboardInterrupt in run %d", run_num)
        logging.error(e)
        torch.cuda.empty_cache()
        return False

    except Exception as e:
        logging.error("Error in training run %d", run_num)
        logging.error(e)
        return False

    return True


def test_model(
    model: ultralytics.models.yolo.model.YOLO, 
    test_args: dict,
    name: str, 
    run_num: int, 
    save=False
) -> bool:
    try:
        t0 = time()
        testing_results: ultralytics.utils.metrics.DetMetrics = model.val(**test_args, name=name)
        testing_time = time() - t0
        
        if testing_results is not None:
            # Get results
            p = testing_results.results_dict["metrics/precision(B)"]
            r = testing_results.results_dict["metrics/recall(B)"]
            f1 = 2 * p * r / (p + r)
            
            # Print results
            logging.info("========== TESTING RESULT FOR %s==========", name)
            logging.info("Time taken: %.2f seconds", testing_time)
            logging.info("Average F1 Score: %.2f", f1)
            logging.info("Average Precision: %.2f", p)
            logging.info("Average Recall: %.2f", r)
            logging.info("Average mAP: %.2f", testing_results.results_dict["metrics/mAP50(B)"])
            logging.info("========== END TESTING RESULT ==========")
            
            # Get the image and labels paths
            if save:
                save_path = P(testing_results.save_dir) / "predictions"
                save_path.mkdir(exist_ok=True)
                with open("./data.yaml", "r") as f:
                    yaml_data = yaml.safe_load(f)
                data_path = P(yaml_data["data"])
                images_path = data_path / "images" / "test"
                labels_path = data_path / "labels" / "test"
                assert images_path.exists(), f"Image path does not exist: {images_path}"
                assert labels_path.exists(), f"Labels path does not exist: {labels_path}"
                
                # Iterate over images, draw boxes on them, and save them
                for image in tqdm(list(images_path.iterdir()), desc="Saving test split prediction images..."):
                    #* Problem is happing here. Something about AutoBatch with batch<1. Doesn't make sense.
                    result = model.predict(str(image.resolve()), verbose=False, conf=0.3, device="0")
                    annotated_frame = result[0]
                    annotated_frame.save(filename=str(save_path / image.name))
                
    except torch.cuda.OutOfMemoryError as e:
        logging.error("CUDA out of memory error in testing run %d", run_num)
        logging.error(e)
        torch.cuda.empty_cache()
        return False
    
    except Exception as e:
        logging.error("Error in testing run %d", run_num)
        logging.error(e)
        torch.cuda.empty_cache()
        return False
    
    return True


def train_test_model(
    model: ultralytics.models.yolo.model.YOLO, 
    training_args: dict, 
    test_args: dict, 
    tuning_space: dict, 
    skip=0, 
    save=True
) -> None:
    grid = list(product(*tuning_space.values(), repeat=1))
    keys = list(tuning_space.keys())
    
    for run_num, pair in list(enumerate(grid))[skip:]:
        torch.cuda.empty_cache()
    
        # Get args for training
        name_parts = [f"{keys[i]}{pair[i]}" for i in range(len(pair))]
        training_args["name"] = f"run{run_num}--" + "_".join(name_parts)
        tuning_args = dict(zip(keys, pair))
    
        # Train and test
        if train_model(model, training_args, tuning_args, run_num):
            test_model(model, test_args, training_args["name"], run_num, name_parts, save=save)
