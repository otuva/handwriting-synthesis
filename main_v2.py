import numpy as np
from handwriting_synthesis import Hand

if __name__ == '__main__':
    # Prompt the user for input
    text = input("Enter the text you want to use: ")

    hand = Hand()

    # Split the input text into sentences with a maximum length of 70 characters
    sentences = []
    current_sentence = ""
    for word in text.split():
        if len(current_sentence) + len(word) <= 70:
            current_sentence += " " + word
        else:
            sentences.append(current_sentence.strip())
            current_sentence = word
    sentences.append(current_sentence.strip())

    # Prompt the user for the bias and style behavior choice
    print("Choose the bias and style behavior:")
    print("1) Fixed bias and fixed style")
    print("2) Variable bias and fixed style")
    print("3) Fixed bias and variable style")
    choice = input("Enter your choice (1/2/3): ")

    # Define the handwriting characteristics based on the user's choice
    biases = []
    styles = []
    # stroke_colors = ['red', 'green', 'black', 'blue']
    # stroke_widths = [1, 2, 1, 2]
    if choice == "1":
        biases = [.95 for _ in sentences]
        styles = [1 for _ in sentences]
    elif choice == "2":
        biases = [.75 for _ in sentences]
        styles = np.cumsum(np.array([len(i) for i in sentences]) == 0).astype(int)
    elif choice == "3":
        biases = .2 * np.flip(np.cumsum([len(i) == 0 for i in sentences]), 0)
        styles = [7 for _ in sentences]
    else:
        print("Invalid choice. Using default behavior.")

    # Generate the handwriting image
    hand.write(
        filename='img/usage_v2demo.svg',
        lines=sentences,
        biases=biases,
        styles=styles,
        # stroke_colors=stroke_colors,
        # stroke_widths=stroke_widths
    )
