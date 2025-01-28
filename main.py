import torch
import clip
from PIL import Image
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import argparse
from wilds.common.data_loaders import get_train_loader
import torchvision.transforms as transforms
from data.Waterbird import WaterbirdsDataset
from data.CelebA import CelebADataset
from utils import classify_images, accuracy_by_subgroup, print_per_class, orth_transforamtion_calculation, train_transformation
from data.SSL_text import get_SSL_dataset
from models import get_transformer
import wandb


def main(args):
    if args.wandb is not None:
        wandb.init(project='CLIPSB',name=args.wandb, config=args)

    #Load CLIP
    model, preprocess = clip.load(args.CLIP_model , device=args.device)


    # Load the full dataset, and download it if necessary
    if args.dataset == 'celeba':
        dataset = CelebADataset(download=True)
    elif args.dataset == 'waterbirds':
        dataset = WaterbirdsDataset(download=True)

    # Define preprocessing transformation
    transform = transforms.Compose([
        preprocess,  # Use CLIP's preprocess (resizes and normalizes the image)
    ])


    # Get the training set
    test_data = dataset.get_subset(
        "test",
        transform=transform,
    )

    # Prepare the standard data loader
    test_loader = get_train_loader("standard", test_data, batch_size=args.batch_size, num_workers=0)

    if args.dataset == 'celeba':
        classes = ["a celebrity with dark hair", "a celebrity with blond hair"]
    elif args.dataset == 'waterbirds':
        classes = [ "Landbird", "Waterbird"]
    # Create prompts for classes
    def create_prompts(class_names):
        templates = [
            "a photo of a {}.",
            "a picture of a {}.",
        ]
        prompts = []
        for cls in class_names:
            for temp in templates:
                prompts.append(temp.format(cls))
        return prompts

    text_prompts = create_prompts(classes)
    print("The classification prompts are:", text_prompts)

    # Tokenize and encode text prompts
    text_tokens = clip.tokenize(text_prompts).to(args.device)
    with torch.no_grad():
        text_embeddings = model.encode_text(text_tokens)
        text_embeddings /= text_embeddings.norm(dim=-1, keepdim=True)

    # Average embeddings per class
    text_embeddings = text_embeddings.view(len(classes), -1, text_embeddings.shape[-1])
    text_embeddings = text_embeddings.mean(dim=1)
    text_embeddings /= text_embeddings.norm(dim=-1, keepdim=True)
    text_embeddings = text_embeddings.to(torch.float32)

    transfromer = None
    P = None

    if args.mitigation is not None:
        if args.mitigation == 'orth':
            if args.dataset == 'celeba':
                spurious_words = ["Man", "Woman"]
            elif args.dataset == 'waterbirds':
                spurious_words = ["water", "land"]

            P = orth_transforamtion_calculation(args, model, spurious_words)
        
        elif args.mitigation == 'train':
            text_loader = get_SSL_dataset(args)
            transformer = get_transformer(args)
            transformer.load_state_dict(torch.load('transformer.pth'))
            # train_transformation(args, model, text_loader, transformer)
            
        
        
    accuracy, misclassified_samples, [all_y, all_preds, all_metadata] = classify_images(args, model, text_embeddings, test_loader, P=P, transformer=transformer, description="Classifying")


    eval_results = dataset.eval(all_preds.to('cpu'), all_y.to('cpu'), all_metadata.to('cpu'))
    # Assuming eval_results is a tuple with a dictionary and a string
    results_dict = eval_results[0]

    # Extract and print the desired metrics
    if args.dataset == 'celeba':
        adj_acc_avg = results_dict['acc_avg']
    elif args.dataset == 'waterbirds':  
        adj_acc_avg = results_dict['adj_acc_avg']
    worst_group_acc = results_dict['acc_wg']

    print(f"Adjusted Average Accuracy: {adj_acc_avg:.3f}")
    print(f"Worst-Group Accuracy: {worst_group_acc:.3f}")
    wandb.log({"Adjusted Average Accuracy": adj_acc_avg, "Worst-Group Accuracy": worst_group_acc})

    if args.per_group:
        print_per_class(args, eval_results)
        accuracy_by_subgroup(list(all_preds.to('cpu').numpy()), list(all_y.to('cpu').numpy()), [x[0] for x in list(all_metadata.to('cpu').numpy())])

if __name__ == "__main__":

    args = argparse.ArgumentParser()
    args.add_argument('--device', type=str.lower, default='cpu')
    args.add_argument('--batch_size', type=int, default=64)
    args.add_argument('--CLIP_model', type=str, default='ViT-L/14@336px',
                    help='CLIP model to use [ViT-B/32, RN50, RN101, RN50x4, ViT-B/16, ViT-L/14@224px, ViT-L/14@336px]')
    args.add_argument('--dataset', type=str.lower, default='waterbirds',
                        help='dataset to use [waterbirds, celeba]')
    args.add_argument('--epochs', type=int, default=5
                        , help='number of epochs to train the embedding transformer')
    args.add_argument('--per_group', type=bool, default=False
                        , help='whether to print accuracy per group')
    args.add_argument('--mitigation', type=str.lower, default=None
                        , help='What mitigation technique to use [None, orth, train]')
    args.add_argument('--num_bases', type=int, default=0
                        , help='Free transformation if zero otherwise number of bases for orthogonalization')
    args.add_argument('--wandb', type=str, default=None
                        , help='wandb run name')

    args = args.parse_args()
    args.device = ['cuda' if torch.cuda.is_available() else 'cpu'][0]

    #print(args)
    print(args)
    print('*'*10)

    main(args)
