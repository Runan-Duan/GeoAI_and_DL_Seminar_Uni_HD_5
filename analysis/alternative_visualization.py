import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import Variable
import torch.nn.functional as F
from torchvision import transforms, models
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader
from torchvision import datasets

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load your trained model 
model = models.resnet18(pretrained=True).to(device)
model.eval()

# Define a simple image transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load a sample image (you can replace this with your image)
image_path = "path_to_your_image.jpg"
img = Image.open(image_path).convert('RGB')
img = transform(img).unsqueeze(0).to(device)  # Add batch dimension

# Forward pass
output = model(img)

# Get the class prediction
_, predicted_class = output.max(1)

# Define a function for Grad-CAM
def generate_gradcam(model, input_image, target_class):
    model.eval()
    input_image.requires_grad_()
    
    # Forward pass
    output = model(input_image)
    class_score = output[0, target_class]

    # Backward pass
    model.zero_grad()
    class_score.backward()

    # Get the gradients and activations
    gradients = input_image.grad[0].cpu().detach().numpy()
    activations = model.layer4[1].conv2.weight[0].cpu().detach().numpy()

    # Generate the Grad-CAM heatmap
    gradcam = np.dot(activations, gradients)
    gradcam = np.maximum(gradcam, 0)  # ReLU
    gradcam = cv2.resize(gradcam, (224, 224))
    
    # Normalize the heatmap
    gradcam = gradcam / gradcam.max()

    return gradcam

# Visualize the Grad-CAM heatmap
def show_gradcam(image, gradcam, title="Grad-CAM"):
    # Convert image to numpy
    img_np = image.squeeze().cpu().numpy().transpose(1, 2, 0)
    img_np = np.clip(img_np, 0, 1)
    
    # Create a figure
    plt.figure(figsize=(10, 5))
    
    # Plot the original image
    plt.subplot(1, 2, 1)
    plt.imshow(img_np)
    plt.title('Original Image')
    
    # Plot the Grad-CAM heatmap
    plt.subplot(1, 2, 2)
    plt.imshow(img_np, alpha=0.6)
    plt.imshow(gradcam, cmap='jet', alpha=0.4)
    plt.title(title)
    plt.show()

# Generate Grad-CAM heatmap
gradcam = generate_gradcam(model, img, predicted_class)

# Show Grad-CAM heatmap
show_gradcam(img, gradcam)

# ---- t-SNE Visualization ----
def visualize_tsne(model, data_loader):
    features = []
    labels = []

    with torch.no_grad():
        model.eval()
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            features.append(output.cpu().numpy())
            labels.append(target.cpu().numpy())
    
    features = np.concatenate(features, axis=0)
    labels = np.concatenate(labels, axis=0)

    # Perform t-SNE
    tsne = TSNE(n_components=2, random_state=42)
    tsne_results = tsne.fit_transform(features)

    # Plot t-SNE results
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=labels, cmap='viridis', s=10)
    plt.colorbar(scatter)
    plt.title("t-SNE Visualization of Feature Embeddings")
    plt.show()

# Example data loader (replace this with your own dataset)
test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Visualize t-SNE on test dataset
visualize_tsne(model, test_loader)
