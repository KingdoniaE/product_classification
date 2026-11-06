from sys import argv

if __name__ == '__main__':
    print(argv)

    if len(argv) != 2:
        print("Usage: python3 main.py path")
    else:
        script = argv[1]
        match script:
            case 'process':
                from src.process.preprocess import preprocess
                preprocess()
            case 'train':
                from src.runner.train import train
                train()
