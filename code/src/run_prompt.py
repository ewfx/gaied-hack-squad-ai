import torch
from transformers import BertTokenizer, BertForSequenceClassification
from readOutlookMails import OutlookMailBox

import csv

def write_dict_to_csv(filename, data_dict_list, fieldnames=None):
    """
    Writes a list of dictionaries (data_dict_list) to a CSV file.

    Args:
        filename: The name of the CSV file to write to (e.g., "output.csv").
        data_dict_list: A list of dictionaries, where each dictionary represents a row.
        fieldnames: An optional list of strings representing the header row (keys of the dictionaries).
                    If None, the keys of the first dictionary are used.
    """
    try:
        with open(filename, 'a', newline='', encoding='utf-8') as csvfile:
            if not data_dict_list: #Handle empty list case
                print(f"Warning: data_dict_list is empty. No CSV file created.")
                return

            if fieldnames is None:
                fieldnames = data_dict_list[0].keys()

            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writeheader()  # Write the header row
            writer.writerows(data_dict_list)  # Write the data rows

        print(f"Data successfully written to {filename}")

    except Exception as e:
        print(f"An error occurred: {e}")

def predict_email_category(email_text):
    model = BertForSequenceClassification.from_pretrained("model_fine_tuned_bert_email")
    tokenizer = BertTokenizer.from_pretrained("model_fine_tuned_bert_email")

    inputs = tokenizer(email_text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
        prediction = torch.argmax(outputs.logits).item()

    label_mapping = {
        0: "AU Transfer",
        1: "Adjustment",
        2: "Closing Notice",
        3: "Commitment Change",
        4: "Money Movement Inbound",
        5: "Money Movement Outbound"
    }

    return label_mapping[prediction]

def read_unread_mails_and_category_it():
    outlook = OutlookMailBox("hack2025")
    emails = outlook.read_mails_from_outlook()

    if emails:
        for email in emails:
            test_email = email["Body"]
            del email["Body"]
            if test_email:
                email["Predicted Category"] = predict_email_category(test_email)

        write_dict_to_csv(filename="Predicted_Category.csv", data_dict_list=emails)


if __name__ == "__main__":
    read_unread_mails_and_category_it()
