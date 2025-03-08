import torch
from torch.utils.data import Dataset, DataLoader


def generate_random_list(n, m, seed=42):
    """
    Generate a list of n integers randomly filled with numbers from [1, m]
    where m < n, using torch with controlled seed.
    
    Args:
        n (int): Number of elements in the list.
        m (int): Upper bound (inclusive) for random numbers.
        seed (int): Random seed for reproducibility.

    Returns:
        list: A list of n integers in range [1, m].
    """
    torch.manual_seed(seed)  # Set the fixed seed for reproducibility
    random_list = torch.randint(0, m , (n,))  # Generate n numbers from [1, m]
    return random_list.tolist()  # Convert to a Python list


class Bird_SSL_Dataset(Dataset):
    def __init__(self, birds, backgrounds, num_samples=100, seed = 42):
        super().__init__()
        self.num_samples = num_samples
        self.birds1 = birds[0]
        self.birds2 = birds[1]

        self.backgrounds1 = backgrounds[0]
        self.backgrounds2 = backgrounds[1]

        # create the shuffled list of birds and backgrounds
        self.birds1_idx = generate_random_list(num_samples, len(self.birds1), seed=seed)
        self.birds2_idx = generate_random_list(num_samples, len(self.birds2), seed=seed)
        self.backgrounds1_idx = generate_random_list(num_samples, len(self.backgrounds1), seed=seed)
        self.backgrounds2_idx = generate_random_list(num_samples, len(self.backgrounds2), seed=seed)
        self.bird1vsbird2 = generate_random_list(num_samples, 2, seed=seed)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Randomly sample a bird and a background
        bird1_idx = self.birds1_idx[idx]
        bird2_idx = self.birds2_idx[idx]
        background1_idx = self.backgrounds1_idx[idx]
        background2_idx = self.backgrounds2_idx[idx]
        
        bird1 = self.birds1[bird1_idx]  
        bird2 = self.birds2[bird2_idx]
        
        background1 = self.backgrounds1[background1_idx]
        background2 = self.backgrounds2[background2_idx]



        # Create the sentence for negative samples
        sentence_neg_1 = f"A photo of a {bird1}"
        sentence_neg_2 = f"A photo of a {bird2}"
        
        if self.bird1vsbird2[idx] == 1:
            pos_bird = bird1
        else:
            pos_bird = bird2

        # create the sentence for positive samples
        sentence_pos_1 = f"A photo of a {pos_bird} in a {background1}."
        sentence_pos_2 = f"A photo of a {pos_bird} in a {background2}."

        sentences = [sentence_neg_1, sentence_neg_2, sentence_pos_1, sentence_pos_2]


        
        return sentences


class Celeb_SSL_Dataset(Dataset):
    def __init__(self, celebs, hair, num_samples=100):
        super().__init__()
        self.num_samples = num_samples
        
        # Combine birds and backgrounds for sampling
        self.celebs1 = celebs[0]
        self.celebs2 = celebs[1]
        self.hair1 = hair[0]
        self.hair2 = hair[1]

        # create the shuffled list of celebrities and hairs
        self.celebs1_idx = generate_random_list(num_samples, len(self.celebs1))
        self.celebs2_idx = generate_random_list(num_samples, len(self.celebs2))
        self.hair1_idx = generate_random_list(num_samples, len(self.hair1))
        self.hair2_idx = generate_random_list(num_samples, len(self.hair2))
        self.hair1vshair2 = generate_random_list(num_samples, 2)

    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # Randomly sample a bird and a background
        celeb1_idx = self.celebs1_idx[idx]
        celeb2_idx = self.celebs2_idx[idx]
        hair1_idx = self.hair1_idx[idx]
        hair2_idx = self.hair2_idx[idx]

        celeb1 = self.celebs1[celeb1_idx]
        celeb2 = self.celebs2[celeb2_idx]

        hair1 = self.hair1[hair1_idx]
        hair2 = self.hair2[hair2_idx]


    
        # Create the sentence for negative samples
        sentence_neg_1 = f"A photo of a person with a {hair1} hair"
        sentence_neg_2 = f"A photo of a person with a {hair2} hair"
        
        if self.hair1vshair2[idx] == 1:
            pos_hair = hair1
        else:   
            pos_hair = hair2
        # create the sentence for positive samples
        sentence_pos_1 = f"A photo of {celeb1} with a {pos_hair} hair."
        sentence_pos_2 = f"A photo of {celeb2} with a {pos_hair} hair."

        sentences = [sentence_neg_1, sentence_neg_2, sentence_pos_1, sentence_pos_2]

        return sentences
    




class Celeb_SSL_Dataset(Dataset):
    def __init__(self, args, scene_descriptions, genders, blond, dark, num_samples=100):
        super().__init__()
        self.num_samples = num_samples
        
        # Combine birds and backgrounds for sampling
        self.sd = scene_descriptions
        self.gender = genders
        self.hair1 = blond
        self.hair2 = dark

        # create the shuffled list of celebrities and hairs
        self.sd_idx = generate_random_list(num_samples, len(self.sd), seed=args.seed)
        self.gender_idx = generate_random_list(num_samples, len(self.gender), seed=args.seed)
        self.hair1_idx = generate_random_list(num_samples, len(self.hair1), seed=args.seed)
        self.hair2_idx = generate_random_list(num_samples, len(self.hair2), seed=args.seed)
        self.hair1vshair2 = generate_random_list(num_samples, 2)


    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # Randomly sample a bird and a background
        scene_idx = self.sd_idx[idx]
        gender_idx = self.gender_idx[idx]
        hair1_idx = self.hair1_idx[idx]
        hair2_idx = self.hair2_idx[idx]

        sentence_neg_1 = self.sd[scene_idx].replace("*Gender*", self.gender[gender_idx]).replace("*hair*", self.hair1[hair1_idx])
        sentence_neg_2 = self.sd[scene_idx].replace("*Gender*", self.gender[gender_idx]).replace("*hair*", self.hair2[hair2_idx])


        
        if self.hair1vshair2[idx] == 1:
            pos_hair = self.hair1[hair1_idx]
        else:   
            pos_hair = self.hair2[hair2_idx]

        # create the sentence for positive samples
        if gender_idx == 0:
            opp_gender = 1
        else:
            opp_gender = 0
        sentence_pos_1 =  self.sd[scene_idx].replace("*Gender*", self.gender[gender_idx]).replace("*hair*", pos_hair)
        sentence_pos_2 =  self.sd[scene_idx].replace("*Gender*", self.gender[opp_gender]).replace("*hair*", pos_hair)

        sentences = [sentence_neg_1, sentence_neg_2, sentence_pos_1, sentence_pos_2]

        return sentences
    

def get_SSL_dataset(args):
    # Define lists of birds and backgrounds


    with open("data/land_birds.txt", "r") as file:
        land_birds = [line.strip() for line in file]

    
    with open("data/water_birds.txt", "r") as file:
        water_birds = [line.strip() for line in file]

    birds = [land_birds , water_birds]

    with open("data/water_backgrounds.txt", "r") as file:
        water_backgrounds = [line.strip() for line in file]
    with open("data/ground_backgrounds.txt", "r") as file:
        ground_backgrounds = [line.strip() for line in file]

    backgrounds = [water_backgrounds , ground_backgrounds]



    with open("data/scene_descriptions.txt", "r") as file:
        scene_descriptions = [line.strip() for line in file]
    with open("data/genders.txt", "r") as file:
        genders = [line.strip() for line in file]
    with open("data/blond_hair_descriptions.txt", "r") as file:
        blond_hair_descriptions = [line.strip() for line in file]
    with open("data/dark_hair_descriptions.txt", "r") as file:
        dark_hair_descriptions = [line.strip() for line in file]
        
    if args.dataset == 'celeba':
        # textset = Celeb_SSL_Dataset(celebs, hair, num_samples=args.num_samples)
        textset = Celeb_SSL_Dataset(args, scene_descriptions, genders, blond_hair_descriptions, dark_hair_descriptions, num_samples=args.num_samples)
        textloader = DataLoader(textset, batch_size=3, shuffle=True)
    elif args.dataset == 'waterbirds':
        textset = Bird_SSL_Dataset( birds, backgrounds, num_samples=args.num_samples, seed=args.seed)
        textloader = DataLoader(textset, batch_size=3, shuffle=True)
    
    print("data example")
    print(textset[0])
    return textloader
