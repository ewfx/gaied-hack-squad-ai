import win32com.client

class OutlookMailBox:

    def __init__(self, folder_name="Inbox"):
        self.folder_name = folder_name


    def get_unread_email_count(self):
        """
        Retrieves the number of unread emails in a specified Outlook folder.

        Returns:
            int: The number of unread emails, or -1 if an error occurs.
        """
        try:
            outlook = win32com.client.Dispatch("Outlook.Application").GetNamespace("MAPI")
            folder = outlook.Folders.Item(1).Folders(self.folder_name)
            unread_count = folder.UnReadItemCount
            return unread_count

        except Exception as e:
            print(f"An error occurred: {e}")
            return -1

    def read_and_mark_emails(self, num_emails=10, unread_only=True):
        """
        Reads Outlook emails from a specified folder and marks them as read.

        Args:
            num_emails (int): The number of emails to retrieve and mark.
            unread_only (bool): If True, only retrieve unread emails.

        Returns:
            list: A list of dictionaries, where each dictionary represents an email.
                  Each dictionary contains keys like "Subject", "Sender", "Body", "ReceivedTime".
                  Returns an empty list if an error occurs or no emails are found.
        """
        try:
            outlook = win32com.client.Dispatch("Outlook.Application").GetNamespace("MAPI")
            folder = outlook.Folders.Item(1).Folders(self.folder_name)
            messages = folder.Items

            email_list = []
            count = 0

            for message in messages:
                if unread_only and not message.UnRead:
                    continue

                email_data = {
                    "Subject": message.Subject,
                    "Sender": message.SenderName,
                    "Body": message.Body,
                    "ReceivedTime": message.ReceivedTime,
                }
                email_list.append(email_data)

                # Mark the email as read
                message.UnRead = False
                message.Save()  # Important to save the change

                count += 1
                if count >= num_emails:
                    break

            return email_list

        except Exception as e:
            print(f"An error occurred: {e}")
            return []

    def read_mails_from_outlook(self):
        unread_count = self.get_unread_email_count()
        if unread_count == 0:
            print(f"No unread email in {self.folder_name} folder.")
            return []
        elif unread_count < 0:
            print("Could not retrieve unread email count.")
            return []
        else:
            print(f"Number of unread emails in {self.folder_name}: {unread_count}")
            return self.read_and_mark_emails(num_emails=unread_count, unread_only=True)

if __name__ == "__main__":
    obj = OutlookMailBox("hack2025")
    emails = obj.read_mails_from_outlook()
    if emails:
        for email in emails:
            print(f"Subject: {email['Subject']}")
            print(f"Sender: {email['Sender']}")
            print(f"Received Time: {email['ReceivedTime']}")
            if email['Body']:
                print(f"Body: {email['Body'][:100]}...")
            else:
                print("Body: (Empty)")
            print("-" * 20)
    else:
        print("No emails found or an error occurred.")
