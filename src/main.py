
from argparse import ArgumentParser
if __name__ == '__main__':
    parser = ArgumentParser(usage='usge: main.py action')
    parser.add_argument('action',choices=['evaluate','train','predict','preprocess','run'])
    args=parser.parse_args()
    action = args.action

    match action:
        case 'train':
            from src.runner.train import train
            train()
        case 'preprocess':
            from src.process.preprocess import preprocess
            preprocess()
        case 'evaluate':
            from src.runner.evaluate import evaluate
            evaluate()
        case 'predict':
            from src.runner.predict import predict
            predict()
        case 'run':
            from src.web.app import run
            run()
