from datasets import load_from_disk, load_dataset, concatenate_datasets
import argparse
import os
import jsonlines

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str)
parser.add_argument('--save_dir', type=str)
parser.add_argument('--doc_column', type=str, default="code")
parser.add_argument('--query_column', type=str, default="docstring")
parser.add_argument('--index_retrieval_ratio', type=float, default=32)
parser.add_argument('--summarization', action="store_true")
parser.add_argument('--train_samples', type=int, default=-1)
parser.add_argument('--test_samples', type=int, default=5000)
args = parser.parse_args()

if os.path.isdir(args.data_path):
    dataset = load_from_disk(args.data_path)
else:
    dataset = load_dataset("json", data_files=args.data_path)


language = args.data_path.split("/")[-1].split(".")[0]
trainset = dataset["train_small"]
testset = dataset["test"]
columns = trainset.column_names

print(len(trainset), len(testset))

if args.summarization:
    trainset = trainset.map(lambda example: {"text_id": example[args.query_column], "text": f"Generate docstring for this {language} code: {example[args.doc_column]}"}).remove_columns(columns)
    testset = testset.map(lambda example: {"text_id": example[args.query_column], "text": f"Generate docstring for this {language} code: {example[args.doc_column]}"}).remove_columns(columns)
    if args.train_samples != -1:
        trainset = trainset.shuffle(seed=42)
        trainset = trainset.take(args.train_samples)
    testset = testset.shuffle(seed=42)
    testset = testset.take(args.test_samples)
    print(len(trainset), len(testset))
    trainset.to_json(os.path.join(args.save_dir, f"train/{language}.json"))
    testset.to_json(os.path.join(args.save_dir, f"test/{language}.json"))
else:
    def process_query(query):
        return "Query: " + " ".join(query.lower().split())

    def process_doc(code):
        return "Code: " + code

    trainset = trainset.map(lambda example: {"query": process_query(example[args.query_column]), "doc": process_doc(example[args.doc_column])})
    testset = testset.map(lambda example: {"query": process_query(example[args.query_column]), "doc": process_doc(example[args.doc_column])})

    trainset = trainset.add_column("text_id", ["train_" + str(i) for i in range(len(trainset))]).remove_columns(columns)
    testset = testset.add_column("text_id", ["test_" + str(i) for i in range(len(testset))]).remove_columns(columns)

    merged_dataset = concatenate_datasets([trainset, testset]).shuffle(seed=42)


    train_retrieval = []
    train_indexing = []
    test_data = []
    cnt = 0
    for dp in merged_dataset:
        if dp["text_id"].startswith("train"):
            train_retrieval.append({"text_id": str(cnt), "text": dp["query"]})
            train_indexing.append({"text_id": str(cnt), "text": dp["doc"]})
        else:
            train_indexing.append({"text_id": str(cnt), "text": dp["doc"]})
            test_data.append({"text_id": str(cnt), "text": dp["query"]})
        cnt += 1
        
    # for dp in testset:
    #     train_indexing.append({"text_id": dp["text_id"], "text": dp["doc"]})
    #     test_data.append({"text_id": dp["text_id"], "text": dp["query"]})
        
    train_retrieval = train_retrieval[:int(len(train_indexing)/args.index_retrieval_ratio)]

    print(len(train_retrieval), len(train_indexing), len(train_indexing)/len(train_retrieval))
    train_data = train_retrieval + train_indexing

    with jsonlines.open(os.path.join(args.save_dir, f"{language}_train_r{args.index_retrieval_ratio}.json"), mode='w') as writer:
        writer.write_all(train_data)
        
    with jsonlines.open(os.path.join(args.save_dir, f"{language}_test_r{args.index_retrieval_ratio}.json"), mode='w') as writer:
        writer.write_all(test_data)