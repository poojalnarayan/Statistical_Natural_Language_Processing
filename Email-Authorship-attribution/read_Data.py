import re
import os

#Patterns to match
id_pat = re.compile("Message-ID.+\n")
to_sub_pat = re.compile("To:([ ,\n\t]|['\.a-z0-9]+@['\.a-z0-9]+)+")
to_pat = re.compile("To:([ ,\n\t])?(.*)+\n")
cc_pat = re.compile("b?cc:.+\n", re.IGNORECASE)
subject_pat = re.compile("Subject:.+\n")
from_pat = re.compile("From:(.*)+\n")
sent_pat = re.compile("Sent:.+\n")
received_pat = re.compile("Received:.+\n")
ctype_pat = re.compile("Content-Type:.+\n")
reply_pat = re.compile("Reply- Organization:.+\n")
date_pat = re.compile("Date:.+\n")
xmail_pat = re.compile("X-Mailer:.+\n")
mimever_pat = re.compile("mime-version:.+\n", re.IGNORECASE)
contentinfo_pat = re.compile("----------------------------------------.+----------------------------------------")
forwardedby_pat = re.compile("----+ Forwarded.+----+.*", re.DOTALL)
caution_pat = re.compile('''\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*.+\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*''')
privacy_pat = re.compile(" _______________________________________________________________.+ _______________________________________________________________")
origmessage_pat = re.compile("-----Original Message-----.*", re.DOTALL)
x_from_pat = re.compile("X-From:(.*)<(.*)>+\n")
x_to_pat = re.compile("X-To:(.*)<(.*)>+\n")
x_pat = re.compile("X-.+\n")
encode_pat = re.compile("Content-Transfer-Encoding.+\n")
attachment_pat = re.compile(" --------- Inline attachment follows ---------.*", re.DOTALL)
word_pat = re.compile("[a-zA-Z]+");

#To store all people from whom he received emails
from_emails_map = {}

root_dir = "/home/pooja/Documents/LING_439_539/project/maildir/taylor-m/" #test_my_code/"
output_dir = "/home/pooja/Documents/LING_439_539/project/from_email_body_extracted/"
for directory, subdirectory, filenames in  os.walk(root_dir):
    if directory.__contains__("sent"):
        continue
    print(directory, subdirectory, filenames)   #here subdirectories are empty mostly
    for filename in filenames:
        with open(os.path.join(directory, filename), "r") as f:
            try:
                data = f.read()
            except UnicodeDecodeError:
                print("Unicode error for file %s_%s" % (directory, filename))
                continue

        print("============",directory,"_",filename, "=============\n")
        if directory.__contains__("notes_inbox") and filename== "307.": #for taylor-m, there is no To: field
            continue
        if directory.__contains__("inbox") and (filename == "285." or filename == "74."):  # for taylor-m, there is To: field broken
            continue
        if directory.__contains__("all_documents") and filename== "1608.": #for taylor-m, there is no To: field
            continue

        emails = data.split(" -----Original Message-----")
        f.close()

        for email in  emails:
            new_text = id_pat.sub('', email)
            #extracting to email, in case of reply and older threads
            if x_to_pat.search(new_text):
                x_to_email = x_to_pat.search(new_text).group(1).strip(' ').replace('\t', '').replace('\n', '')
            else:
                x_to_email = ""
            to_email = to_pat.search(new_text).group().strip(' To:').replace('\t', '').replace('\n', '')
            if x_to_email == "":
                new_text = to_pat.sub('', new_text)
            else:
                new_text = to_sub_pat.sub('', new_text)

            new_text = cc_pat.sub('', new_text)
            new_text = subject_pat.sub('', new_text)

            if x_from_pat.search(new_text):
                x_from_email = x_from_pat.search(new_text).group(1).strip(' ').replace('\t', '').replace('\n', '')
            else:
                x_from_email = ""

            from_email = from_pat.search(new_text).group().strip(' From:').replace('\t', '').replace('\n', '').strip(' ')
            new_text = from_pat.sub('', new_text)

            new_text = sent_pat.sub('', new_text)
            new_text = received_pat.sub('', new_text)
            new_text = ctype_pat.sub('', new_text)
            new_text = reply_pat.sub('', new_text)
            new_text = date_pat.sub('', new_text)
            new_text = xmail_pat.sub('', new_text)
            new_text = mimever_pat.sub('', new_text)
            new_text = contentinfo_pat.sub('', new_text)
            new_text = forwardedby_pat.sub('', new_text)
            new_text = caution_pat.sub('', new_text)
            new_text = privacy_pat.sub('', new_text)
            new_text = origmessage_pat.sub('', new_text)
            new_text = x_pat.sub('', new_text)
            new_text = encode_pat.sub('', new_text)
            new_text = attachment_pat.sub('', new_text)

            # discard file if email doesn't contain any content after filtering
            if not word_pat.search(new_text):
                continue

            #discard the announcements
            if (from_email.__contains__("Announcements") or from_email.__contains__("announcements") or x_from_email.__contains__("Announcements") or x_from_email.__contains__("announcements")):
                continue

            if from_email.__contains__("/") or x_from_email.__contains__("/"):
                continue

            if (x_to_email == ""):
                print("to_email = ", to_email)
            else:
                print("x_to_email = ", x_to_email)

            if(x_from_email == ""):
                if(from_email.__contains__("Allen, Phillip K.")):    #To-Do replace by the specific user inbox
                    print(" Its from same person ignore")
                else:
                    print("from_email = ", from_email)
                    from_emails_map[from_email] = from_emails_map.get(from_email, 0) + 1
                    f = open(os.path.join(output_dir+"taylor-m/",from_email + "_" + str(from_emails_map[from_email])), 'w')
                    f.write(new_text)
                    f.close()
            else:
                print("x_from_email = ", x_from_email)
                from_emails_map[x_from_email] = from_emails_map.get(x_from_email, 0) + 1
                f = open(os.path.join(output_dir+"taylor-m/", x_from_email + "_" + str(from_emails_map[x_from_email])), 'w')
                f.write(new_text)
                f.close()
            #print("==========================================================") from_emails_extracted
            print(new_text)
            print("==========================================================")

print(from_emails_map)


