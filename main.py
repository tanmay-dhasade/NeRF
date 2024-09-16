import argparse
import train

def main():
    parser = argparse.ArgumentParser(description='Train Nerf model')
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for training')
    parser.add_argument('--save_interval', type=int, default=10, help='Interval for saving model checkpoints')
    
    args = parser.parse_args()
    train.train_model(num_epochs=args.num_epchs,
                      save_interval=args.save_interval,
                      batch_size=args.batch_size
    )

if __name__ == "__main__":
    main()