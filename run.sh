BASELINE_FOLDER=unrestricted-advex/unrestricted_advex/mnist_baselines

CUDA_VISIBLE_DEVICES=0 python3 $BASELINE_FOLDER/train_two_class_mnist.py --total_batches 1000
CUDA_VISIBLE_DEVICES=0 python3 $BASELINE_FOLDER/evaluate_two_class_mnist.py

CUDA_VISIBLE_DEVICES=0 python3 $BASELINE_FOLDER/train_two_class_mnist_blur.py --total_batches 1000
CUDA_VISIBLE_DEVICES=0 python3 $BASELINE_FOLDER/evaluate_two_class_mnist_blur.py
