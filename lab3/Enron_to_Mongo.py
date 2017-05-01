import os
import datetime
from pymongo import MongoClient

__author__ = 'k0emt'
MAIL_DIR_PATH = '/Users/zhangyangzuo/Downloads/big data/lab3/maildir'
PREFIX_TRIM_AMOUNT = len(MAIL_DIR_PATH) + 1
MAX_USER_RUN_LIMIT = 50
MAX_USER_EMAILS_PER_FOLDER_FILE_LIMIT = 2
counter = 1


def get_file_contents(file_to_open_name):
    data_file = open(file_to_open_name)
    file_contents = ""
    try:
        for data_line in data_file:
            file_contents += data_line

    finally:
        data_file.close()
    return file_contents.decode('cp1252')


def save_to_database(mailbox_owner_name, sub_folder, file_name, message_contents):
    document = {"mailbox": mailbox_owner_name,
                "subFolder": sub_folder,
                "filename": file_name,
                "contents": message_contents.encode('utf-8')}

    messages = db.messages
    messages.insert(document)
    return

cn = MongoClient('localhost:27017')
db = cn.enron_mail
print("database initialized {0}".format(datetime.datetime.now()))

# all the mail folders
user_counter = 0
previous_owner = ""

for root, dirs, files in os.walk(MAIL_DIR_PATH, topdown=False):
    directory = root[PREFIX_TRIM_AMOUNT:]

    # extract mail box owner
    parts = directory.split('/', 1)
    mailbox_owner = parts[0]

    if previous_owner != mailbox_owner:
        previous_owner = mailbox_owner
        user_counter += 1

    # sub-folder info
    if 2 == len(parts):
        subFolder = parts[1]
    else:
        subFolder = ''

    # files in each mail folder
    folder_email_counter = 0

    for file in files:

        # get the file contents
        name_of_file_to_open = "{0}/{1}".format(root, file)
        contents = get_file_contents(name_of_file_to_open)
        save_to_database(mailbox_owner, subFolder, file, contents)

        folder_email_counter += 1
        counter += 1
        if counter % 100 == 0:
            print("{0} {1}".format(counter, datetime.datetime.now()))

        if MAX_USER_EMAILS_PER_FOLDER_FILE_LIMIT > 0 and MAX_USER_EMAILS_PER_FOLDER_FILE_LIMIT == folder_email_counter:
            break

    if MAX_USER_RUN_LIMIT > 0 and MAX_USER_RUN_LIMIT == user_counter:
        print("Maximum users limit {0} met.".format(MAX_USER_RUN_LIMIT))
        break

db.close
print("database closed {0}".format(datetime.datetime.now()))
print("{0} total records processed".format(counter - 1))