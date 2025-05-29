import cv2, numpy as np
from PIL import Image
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit_aer import AerSimulator
import matplotlib.pyplot as plt
import time

# Convert lincon.txt to grayscale image (do this once)
img = np.array([[ord(c)-48 if c.isdigit() else ord(c)-55 for c in line.strip()]
                for line in open("lincon.txt")], dtype=np.uint8) * 7
img = cv2.resize(img, (128, 128), interpolation=cv2.INTER_NEAREST)
cv2.imwrite("lincoln_gray_128.tiff", img)

# Helper: Integer to bit list
def int_to_bits(value, num_bits):
    return [int(bit) for bit in bin(value)[2:].zfill(num_bits)]

# Quantum grayscale negation
def negate_image_quantum(image_path):
    img = Image.open(image_path).convert("L").resize((100, 100))
    width, height = img.size
    intensity_bits = 8

    matrix = [[img.getpixel((c, r)) for c in range(width)] for r in range(height)]
    classical_matrix = [[255 - matrix[r][c] for c in range(width)] for r in range(height)]
    negated_matrix = [[0 for _ in range(width)] for _ in range(height)]

    input_circuits = []
    processed_circuits = []

    backend = AerSimulator()

    start_time = time.time()

    count_printed = 0
    for r in range(height):
        for c in range(width):
            qr = QuantumRegister(intensity_bits, 'q')
            cr = ClassicalRegister(intensity_bits, 'c')
            qc = QuantumCircuit(qr, cr)

            bits = int_to_bits(matrix[r][c], intensity_bits)
            for j in range(intensity_bits):
                if bits[intensity_bits - 1 - j] == 1:
                    qc.x(qr[j])
            qc.barrier()
            for j in range(intensity_bits):
                qc.x(qr[j])
            qc.barrier()
            qc.measure(qr, cr)

            transpiled = backend.run(qc, shots=1)
            counts = transpiled.result().get_counts()
            bitstring = list(counts.keys())[0]
            value = int(bitstring, 2)
            negated_matrix[r][c] = value

            if count_printed < 5:
                input_qc = QuantumCircuit(qr, cr)
                for j in range(intensity_bits):
                    if bits[intensity_bits - 1 - j] == 1:
                        input_qc.x(qr[j])
                input_qc.barrier()
                input_circuits.append(input_qc)
                processed_circuits.append(qc)
                count_printed += 1

    end_time = time.time()
    execution_time = end_time - start_time

    # Create quantum negated image
    neg_img = Image.new("L", (width, height))
    for r in range(height):
        for c in range(width):
            neg_img.putpixel((c, r), negated_matrix[r][c])

    mse = np.mean((np.array(negated_matrix) - np.array(classical_matrix)) ** 2)
    return img, neg_img, input_circuits, processed_circuits, mse, execution_time

# Main
if __name__ == "__main__":
    image_path = "lincoln_gray_128.tiff"
    result = negate_image_quantum(image_path)

    if result:
        orig_img, neg_img, input_circuits, processed_circuits, mse, exec_time = result

        # Print circuits for first 5 pixels
        for i in range(len(input_circuits)):
            print(f"\nInput Circuit for Pixel {i+1}:")
            print(input_circuits[i].draw(output="text"))
            print(f"\nProcessed Circuit for Pixel {i+1}:")
            print(processed_circuits[i].draw(output="text"))

        # Print performance
        print(f"\n✅ MSE between Quantum and Classical: {mse:.6f}")
        print(f"⏱️ Execution Time: {exec_time:.4f} seconds")

        # Show images side-by-side
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(orig_img.resize((128, 128)), cmap='gray')
        plt.title("Original Image")
        plt.axis("off")

        plt.subplot(1, 2, 2)
        plt.imshow(neg_img.resize((128, 128)), cmap='gray')
        plt.title("Quantum Negated Image")
        plt.axis("off")

        plt.tight_layout()
        plt.show()
