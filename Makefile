TRAIN_TARGET_EXPERIMENT_DIR=data/target_train_experiment/
TEST_TARGET_EXPERIMENT_DIR=data/target_dev_experiment/
TRAIN_NON_TARGET_EXPERIMENT_DIR=data/non_target_train_experiment/
TEST_NON_TARGET_EXPERIMENT_DIR=data/non_target_dev_experiment/

TRAIN_TARGET_DIR=data/target_train/
TEST_TARGET_DIR=data/target_dev/
TRAIN_NON_TARGET_DIR=data/non_target_train/
TEST_NON_TARGET_DIR=data/non_target_dev/

OPT=-v

all: test

test:
	python3 src/main.py $(TEST_NON_TARGET_DIR)

train:
	python3 src/main.py $(TRAIN_TARGET_DIR) $(TEST_TARGET_DIR) $(TRAIN_NON_TARGET_DIR) $(TEST_NON_TARGET_DIR)

experiment:
	python3 src/main.py $(OPT) $(TRAIN_TARGET_EXPERIMENT_DIR) $(TEST_TARGET_EXPERIMENT_DIR) $(TRAIN_NON_TARGET_EXPERIMENT_DIR) $(TEST_NON_TARGET_EXPERIMENT_DIR)
