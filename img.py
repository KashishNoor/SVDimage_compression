import gradio as gr
import numpy as np
import cv2
from PIL import Image
import io
import time

def svd_compress(image, num_singular_values):
    """Convert image to grayscale, apply SVD, and reconstruct using top singular values."""
    if image is None:
        return None, "No image uploaded"
    
    start_time = time.time()  # Measure execution time

    # Convert image to grayscale
    gray_image = image.convert("L")  # "L" mode is grayscale
    gray_np = np.array(gray_image, dtype=np.float32)

    # Compute original image size
    original_buffer = io.BytesIO()
    image.save(original_buffer, format="JPEG", quality=100)
    original_size = original_buffer.getbuffer().nbytes
    original_size_text = f"{original_size / 1024:.2f} KB" if original_size < 1048576 else f"{original_size / (1024 * 1024):.2f} MB"

    # Perform Singular Value Decomposition (SVD)
    try:
        U, S, Vt = np.linalg.svd(gray_np, full_matrices=False)  
    except np.linalg.LinAlgError as e:
        return None, f"Error in SVD Computation: {str(e)}"

    # Keep only top 'num_singular_values' singular values
    S_k = np.zeros_like(S)
    S_k[:int(num_singular_values)] = S[:int(num_singular_values)]

    # Reconstruct the image using selected singular values
    compressed_image = np.dot(U, np.dot(np.diag(S_k), Vt))
    compressed_image = np.clip(compressed_image, 0, 255).astype(np.uint8)  # Ensure valid pixel values
    
    # Convert back to PIL Image
    compressed_pil = Image.fromarray(compressed_image)

    # Compute compression size
    buffer = io.BytesIO()
    compressed_pil.save(buffer, format="JPEG", quality=85)  # Save compressed version
    buffer.seek(0)
    compressed_size = buffer.getbuffer().nbytes
    compressed_size_text = f"{compressed_size / 1024:.2f} KB" if compressed_size < 1048576 else f"{compressed_size / (1024 * 1024):.2f} MB"

    # Execution time
    execution_time = time.time() - start_time

    return compressed_pil, f"Original Size: {original_size_text}\nCompressed Size: {compressed_size_text}\nSingular Values Used: {num_singular_values}\nImage Rank: {np.linalg.matrix_rank(gray_np)}\nProcessing Time: {execution_time:.2f} sec"

# Gradio Interface
iface = gr.Interface(
    fn=svd_compress,
    inputs=[
        gr.Image(type="pil", label="Upload an HD Image"),
        gr.Slider(minimum=1, maximum=100, value=50, label="Number of Singular Values to Keep")  
    ],
    outputs=[
        gr.Image(type="pil", label="Compressed Image"),
        gr.Textbox(label="Compression & SVD Info")
    ],
    title="SVD-Based Image Compression",
    description="Upload an HD image, convert to grayscale, apply Singular Value Decomposition (SVD), and reconstruct using top singular values."
)

if __name__ == "__main__":
    iface.launch()
