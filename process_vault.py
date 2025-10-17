from datasets import load_from_disk, load_dataset, concatenate_datasets
import argparse
import os
import jsonlines
# from utils import get_node_by_kind, lang_parser

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str)
parser.add_argument('--save_dir', type=str)
parser.add_argument('--doc_column', type=str, default="code")
parser.add_argument('--query_column', type=str, default="docstring")
parser.add_argument('--index_retrieval_ratio', type=float, default=32)
parser.add_argument('--summarization', action="store_true")
parser.add_argument('--train_samples', type=int, default=-1)
parser.add_argument('--test_samples', type=int, default=5000)
parser.add_argument('--track_metadata', action="store_true")
args = parser.parse_args()

if os.path.isdir(args.data_path):
    dataset = load_from_disk(args.data_path)
else:
    dataset = load_dataset("json", data_files=args.data_path)

if not os.path.exists(args.save_dir):
    os.mkdir(args.save_dir)

language = args.data_path.split("/")[-1].split(".")[0]
trainset = dataset["train_small"]
testset = dataset["test"]
columns = trainset.column_names

keep_metadata = []
if args.track_metadata:
    keep_metadata = ['hexsha', 'repo', 'path', 'identifier', 'parameters']
    columns = [x for x in columns if x not in keep_metadata]

stop_words = []
def get_identifier(code, max_iden=10):
    code_encode = bytes(code, "utf8")
    root = lang_parser.parse(code_encode)
    root_node = root.root_node

    # print(root_node.children[0].children[1].children[0].children[0].children[1].children[1].children)
    identifiers = get_node_by_kind(root_node, kind=["identifier"])
    identifiers = [x.text.decode() for x in identifiers]
    finalize_iden = []
    for id in identifiers:
        if id in stop_words or id in finalize_iden or len(id) <= 1:
            continue
        finalize_iden.append(id)
    return finalize_iden[:max_iden]

print(len(trainset), len(testset))

if args.summarization:
    trainset = trainset.map(lambda example: {"text_id": example[args.query_column], "text": f"Generate docstring for this {language} code: {example[args.doc_column]}"}).remove_columns(columns)
    testset = testset.map(lambda example: {"text_id": example[args.query_column], "text": f"Generate docstring for this {language} code: {example[args.doc_column]}"}).remove_columns(columns)
    if args.train_samples != -1:
        trainset = trainset.shuffle(seed=42)
        trainset = trainset.take(args.train_samples)
    if args.test_samples != -1:
        testset = testset.shuffle(seed=42)
        testset = testset.take(args.test_samples)
    print(len(trainset), len(testset))
    trainset.to_json(os.path.join(args.save_dir, f"train/vault_{language}.json"))
    testset.to_json(os.path.join(args.save_dir, f"test/vault_{language}.json"))
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
    text_based_id_pool = []
    url_based_id_pool = []
    identifier_based_id_pool= []
    identifier_based_id_pool_dict= {}
    cnt = 0
    for dp in merged_dataset:
        dp_r = {x: dp[x] for x in keep_metadata}
        dp_r["text_id"]= str(cnt)
        text_based_id = "{}({})|{}|{}".format(dp["identifier"], ",".join(x["param"] for x in dp["parameters"]), dp["repo"], dp["path"])
        
        if text_based_id in text_based_id_pool: #remove duplicate function
            continue
        text_based_id_pool.append(text_based_id)
        
        dp_r["url_based_id"] = "/".join([dp["repo"], dp["path"], dp["identifier"] + "(" + ",".join(x["param"] for x in dp["parameters"])+ ")" ])
        # dp_r["identifier_based_id"] = " ".join([dp["repo"].split("/")[1], dp["path"].split("/")[-1].split(".")[0]] + get_identifier(dp["doc"][5:].strip()))
        
        
        assert dp_r["url_based_id"] not in url_based_id_pool, dp_r["url_based_id"]
        url_based_id_pool.append( dp_r["url_based_id"])
        
        # if dp_r["identifier_based_id"] not in identifier_based_id_pool_dict:
        #     identifier_based_id_pool_dict[dp_r["identifier_based_id"]] = 0
        # else:
        #     identifier_based_id_pool_dict[dp_r["identifier_based_id"]] += 1
        #     dp_r["identifier_based_id"] = dp_r["identifier_based_id"] + " {}".format(str(identifier_based_id_pool_dict[dp_r["identifier_based_id"]]))
        
        # assert dp_r["identifier_based_id"] not in identifier_based_id_pool, dp_r["identifier_based_id"]
        # identifier_based_id_pool.append( dp_r["identifier_based_id"])
        
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

    with jsonlines.open(os.path.join(args.save_dir, f"{language}_train_r{args.index_retrieval_ratio}.json"), mode='w') as writer:
        writer.write_all(train_data)
        
    with jsonlines.open(os.path.join(args.save_dir, f"{language}_test_r{args.index_retrieval_ratio}.json"), mode='w') as writer:
        writer.write_all(test_data)
