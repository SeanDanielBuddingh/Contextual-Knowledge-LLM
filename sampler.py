import pandas as pd

# For qualitative eval of short text

filepaths = ['mistral/data/preprocessing_inference/preprocessed_inference_prompt0.csv',
             'mistral/data/preprocessing_inference/preprocessed_inference_prompt1.csv',
             'mistral/data/preprocessing_inference/preprocessed_inference_prompt2.csv'
            ]

# Loop through each file
for filepath in filepaths:
    # Read the CSV file
    df = pd.read_csv(filepath)
    
    # Sample 10 random rows
    sample_df = df.sample(n=10)
    
    # Print the row number (index) and the row
    for index, row in sample_df.iterrows():
        print(f"Row number: {index}")
        print(row)
        print("\n")