from datasets import load_from_disk, load_dataset, concatenate_datasets
import argparse
import os
import jsonlines

prefixes = {
    "apps": "Generate description for this Python code:\n",
    "cosqa": "What is the purpose of below code:\n",
    "synthetic-text2sql": "Generate query for this sql:\n"
}

parser = argparse.ArgumentParser()
parser.add_argument('--data_type', type=str, choices=["apps", "cosqa", "synthetic-text2sql"])
parser.add_argument('--save_dir', type=str)
parser.add_argument('--doc_column', type=str, default="code")
parser.add_argument('--query_column', type=str, default="docstring")
parser.add_argument('--index_retrieval_ratio', type=float, default=32)
parser.add_argument('--summarization', action="store_true")
parser.add_argument('--train_samples', type=int, default=-1)
parser.add_argument('--test_samples', type=int, default=5000)
parser.add_argument('--track_metadata', action="store_true")
args = parser.parse_args()

data_path = f"CoIR-Retrieval/{args.data_type}-queries-corpus"
dataset = load_dataset(data_path)

corpus = dataset["corpus"].add_column("query", dataset["queries"]["text"]).add_column("qid", dataset["queries"]["_id"])
for dp in corpus:
    assert dp["_id"][1:] == dp["qid"][1:]

columns = corpus.column_names

if "text" in columns:
    corpus = corpus.rename_column("text", args.doc_column)
    
if "query" in columns:
    corpus = corpus.rename_column("query", args.query_column)

columns = corpus.column_names
trainset = corpus.filter(lambda x: x["partition"] == "train")
testset = corpus.filter(lambda x: x["partition"] == "test")


keep_metadata = []
if args.track_metadata:
    keep_metadata = ['meta_information']
    columns = [x for x in columns if x not in keep_metadata]


print(len(trainset), len(testset))

if not os.path.exists(args.save_dir):
    os.mkdir(args.save_dir)

if args.summarization:
    trainset = trainset.map(lambda example: {"text_id": example[args.query_column], "text": prefixes[args.data_type] + example[args.doc_column]}).remove_columns(columns)
    testset = testset.map(lambda example: {"text_id": example[args.query_column], "text": prefixes[args.data_type] + example[args.doc_column]}).remove_columns(columns)
    if args.train_samples != -1 and len(trainset) > args.train_samples:
        trainset = trainset.shuffle(seed=42)
        trainset = trainset.take(args.train_samples)
    if args.test_samples != -1 and len(testset) > args.test_samples:
        testset = testset.shuffle(seed=42)
        testset = testset.take(args.test_samples)
    print(len(trainset), len(testset))
    trainset.to_json(os.path.join(args.save_dir, f"train/coir_{args.data_type}.json"))
    testset.to_json(os.path.join(args.save_dir, f"test/coir_{args.data_type}.json"))
else:
    def process_query(query):
        return "Query: " + query

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
    text_based_id_pool = []
    cnt = 0
    for dp in merged_dataset:
        dp_r = {x: dp[x] for x in keep_metadata}
        dp_r["text_id"]= str(cnt)
        # dp_r["text_based_id"] = "{}({})|{}|{}".format(dp["identifier"], ",".join(x["param"] for x in dp["parameters"]), dp["repo"], dp["path"])
        # if dp_r["text_based_id"] in text_based_id_pool: #remove duplicate function
        #     continue
        # text_based_id_pool.append(dp_r["text_based_id"])
        
        if dp["text_id"].startswith("train"):
            dp_r["text"]= dp["query"]
            train_retrieval.append(dp_r)
            
            dp_q = dp_r.copy()
            dp_q["text"]= dp["doc"]
            train_indexing.append(dp_q)
        else:            
            dp_r["text"]= dp["query"]
            test_data.append(dp_r)
            
            dp_q = dp_r.copy()
            dp_q["text"]= dp["doc"]
            train_indexing.append(dp_q)
        cnt += 1
        
    # for dp in testset:
    #     train_indexing.append({"text_id": dp["text_id"], "text": dp["doc"]})
    #     test_data.append({"text_id": dp["text_id"], "text": dp["query"]})
        
    train_retrieval = train_retrieval[:int(len(train_indexing)/args.index_retrieval_ratio)]

    print(len(train_retrieval), len(train_indexing), len(train_indexing)/len(train_retrieval))
    train_data = train_retrieval + train_indexing

    with jsonlines.open(os.path.join(args.save_dir, f"{args.data_type}_train_r{args.index_retrieval_ratio}.json"), mode='w') as writer:
        writer.write_all(train_data)
        
    with jsonlines.open(os.path.join(args.save_dir, f"{args.data_type}_test_r{args.index_retrieval_ratio}.json"), mode='w') as writer:
        writer.write_all(test_data)