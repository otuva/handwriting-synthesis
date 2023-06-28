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

    # Define the handwriting characteristics
    biases = [.75 for _ in sentences]
    styles = [9 for _ in sentences]
    # stroke_colors = ['red', 'green', 'black', 'blue']
    # stroke_widths = [1, 2, 1, 2]

    # Generate the handwriting image
    hand.write(
        filename='img/usage_demo.svg',
        lines=sentences,
        biases=biases,
        styles=styles,
        # stroke_colors=stroke_colors,
        # stroke_widths=stroke_widths
    )
