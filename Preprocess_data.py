import pandas as pd

def process_csv(input_file, output_file):
    # Read the input CSV file into a DataFrame, specifying the encoding
    df = pd.read_csv(input_file, encoding='latin1')

    # Initialize an empty list to store processed data
    processed_data = []

    # Iterate through each row of the DataFrame
    for index, row in df.iterrows():
        # Process the first column
        if pd.notnull(row[0]):
            # Split the text in the first column by whitespace
            words = row[0].split()

            # Add each word as a separate row, with the corresponding value from the second column
            for word in words:
                processed_data.append([word, row[1]])
        else:
            # Skip empty rows
            continue
    # Create a new DataFrame from the processed data
    processed_df = pd.DataFrame(processed_data, columns=['text', 'label'])

    # Write the processed DataFrame to the output file
    processed_df.to_csv(output_file, index=False)

# Example usage
input_file = r"C:\Users\Lenovo\Documents\Semesters\Semester 6\GT\Project\New folder\scraped_data006.csv"
output_file = r"C:\Users\Lenovo\Documents\Semesters\Semester 6\GT\Project\scraped_data006_preprocessed.csv"

process_csv(input_file, output_file)