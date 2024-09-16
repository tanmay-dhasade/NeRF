import os
import torch
from torch.utils.data import DataLoader
from models.network import NeRF
from utils.dataset import NeRFDataset
from utils.rays import compute_rays, position_encoding, render
from tqdm import tqdm
import matplotlib.pyplot as plt

def train_model(num_epochs, save_interval, batch_size):
    dataset = NeRFDataset()
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(dataset)

    model = NeRF()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    print(device)

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    checkpoint_dir = 'checkpoints'
    os.makedirs(checkpoint_dir, exist_ok=True)  # Create directory if it doesn't exist

    plot_dir = 'plots'
    os.makedirs(plot_dir, exist_ok=True)        # Create plots directory if it doesn't exist


    best_loss = float('inf')  # Initialize the best loss to infinity
    epoch_losses = []  # List to store loss per epoch
    test_loss = []
    best_model = None
    for epoch in tqdm(range(num_epochs)):
        running_loss = 0.0
        for batch_idx, (img, pose, focal) in enumerate(dataloader):
            img, pose, focal = img.to(device), pose.to(device), focal.to(device)
            optimizer.zero_grad()

            # Compute rays and forward pass
            ray_direction, ray_origins, depth_vals, query_points = compute_rays(img, pose, focal, 2, 6, 32, device)
            output = model(position_encoding(query_points.view(-1, query_points.shape[-1]), 6))
            output, _, _ = render(output, query_points, ray_origins, depth_vals)

            # Compute loss and backpropagate
            loss = criterion(output, img)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if batch_idx % 10 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Step [{batch_idx}/{len(dataloader)}], Loss: {loss.item()}")

        # Calculate average loss for this epoch
        epoch_loss = running_loss / len(dataloader)
        epoch_losses.append(epoch_loss)

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss}")

        # Save model checkpoint at every 10th epoch or if the loss decreases
        if (epoch + 1) % save_interval == 0 or epoch_loss < best_loss:
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                print(f"New best loss: {best_loss} at epoch {epoch+1}")
                best_model = model
            epoch_loss *= 100
            checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch+1}_{epoch_loss:.3f}.pth')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': epoch_loss,
            }, checkpoint_path)
            runn_loss = 0
            with torch.no_grad():
                for batch, (img, pose, focal) in enumerate(test_dataloader):
                    ray_direction, ray_origins, depth_vals, query_points = compute_rays(img, pose, focal, 2, 5, 10, device)
                    output = best_model(position_encoding(query_points.view(-1, query_points.shape[-1]), 6))
                    output, _, _ = render(output, query_points, ray_origins, depth_vals)

                    # Compute loss and backpropagate
                    loss = criterion(output, img)
                    runn_loss += loss.item()
                e_loss = runn_loss / len(test_dataloader)
                test_loss.append(e_loss)
                print(f"Epoch [{epoch+1}/{num_epochs}], Testing loss: {e_loss}")
        
            print(f"Model saved at epoch {epoch+1} to {checkpoint_path}")
        model = best_model
    plt.plot(range(1, num_epochs+1), epoch_losses, label='Training Loss')
    plt.plot(range(1, len(test_loss)+1), test_loss, label='Testing Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.legend()
    plt.grid(True)
    plt.savefig(plot_path)
    plot_path = os.path.join(plot_dir, f'Training_loss_curve{num_epochs}_{epoch_losses[-1]:.3f}.png') 

    plt.show()  

    print(f"Loss curve saved to {plot_path}")
    
if __name__ == "__main__":
    train_model()
