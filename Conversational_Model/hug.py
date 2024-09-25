#import datetime
#PATH = "M:/AI_DataSets/Discord/Naumberg/dies_das.txt"
#
#with open(PATH, 'r', encoding='utf-8') as f:
#    data = f.readlines()
#
#class ExData:
#    def __init__(self, date, User, Has_Attachmend, Content) -> None:
#        self.date = date
#        self.User = User
#        self.Has_Attachmend = Has_Attachmend
#        self.Content = Content
#        pass
#
#formated_data:list[ExData] = list[ExData]()
#for d in data:
#    splitted = d.split(",")
#    formated_data.append(ExData(splitted[0], splitted[1], splitted[2], splitted[3]))
#
#timed_dict:dict[datetime.datetime, list[ExData]] = dict[datetime.datetime, list[ExData]]()
#max_time_diff = 60
#last_time:datetime.datetime
#for value in formated_data:
#    time = value.date
#    last_time = time
#    if time:
#        timed_dict[time].append(value)
#
#
#print("Done")

from convokit import Corpus, download

FILE_PATH = "M:/AI_DataSets/cornell_movie_dialog/movie-corpus"

#download("movie-corpus", data_dir=FILE_PATH)

corpus = Corpus(filename=FILE_PATH)

conversation_pairs = []
c = 1
for conversation in corpus.iter_conversations():
    utterances = list(conversation.iter_utterances())

    print(f'conversation: {c}')
    c += 1

    for i in range(len(utterances)-1):
        text = utterances[i].text
        response = utterances[i +1].text

        conversation_pairs.append([i, text, response])
        print(f'utterance: {i}')


import csv
with open("M:/AI_DataSets/cornell_movie_dialog/corpus.csv", 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['index', 'text', 'response'])
    writer.writerows(conversation_pairs)


corpus.print_summary_stats()