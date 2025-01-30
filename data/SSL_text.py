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


# class Bird_SSL_Dataset(Dataset):
#     def __init__(self, birds, backgrounds, num_samples=100):
#         super().__init__()
#         self.num_samples = num_samples
        
#         # Combine birds and backgrounds for sampling
#         self.birds = birds[0] + birds[1]
#         self.backgrounds = backgrounds[0] + backgrounds[1]
        
#         # Create bird type labels
#         self.bird_type_labels = {bird: 0 for bird in birds[0]}  # 0 for land birds
#         self.bird_type_labels.update({bird: 1 for bird in birds[1]})  # 1 for water birds
        
#         # Create background type labels
#         self.background_type_labels = {bg: 0 for bg in backgrounds[1]}  # 0 for land background
#         self.background_type_labels.update({bg: 1 for bg in backgrounds[0]})  # 1 for water background
    
#     def __len__(self):
#         return self.num_samples
    
#     def __getitem__(self, idx):
#         # Randomly sample a bird and a background
#         bird1 = self.birds[torch.randint(len(self.birds), (1,)).item()]
#         background1 = self.backgrounds[torch.randint(len(self.backgrounds), (1,)).item()]

#         bird2 = self.birds[torch.randint(len(self.birds), (1,)).item()]
#         while bird2 == bird1:
#             bird2 = self.birds[torch.randint(len(self.birds), (1,)).item()]
        
#         background2 = self.backgrounds[torch.randint(len(self.backgrounds), (1,)).item()]
#         while background2 == background1:
#             background2 = self.backgrounds[torch.randint(len(self.backgrounds), (1,)).item()]

        

#         # Create the sentence for negative samples
#         sentence_neg_1 = f"A photo of a {bird1}"
#         sentence_neg_2 = f"A photo of a {bird2}"
        
#         # create the sentence for positive samples
#         sentence_pos_1 = f"A photo of a {bird1} in a {background1}."
#         sentence_pos_2 = f"A photo of a {bird1} in a {background2}."

#         sentences = [sentence_neg_1, sentence_neg_2, sentence_pos_1, sentence_pos_2]


        
#         return sentences

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
        self.celebs = celebs[0] + celebs[1]
        self.hair = hair[0] + hair[1]
        
        # Create bird type labels
        self.hair_type_labels = {h: 0 for h in hair[0]}  # 0 for land birds
        self.hair_type_labels.update({h: 1 for h in hair[1]})  # 1 for water birds
        
        # Create background type labels
        self.celebs_type_labels = {c: 0 for c in celebs[0]}  # 0 for land background
        self.celebs_type_labels.update({c: 1 for c in celebs[1]})  # 1 for water background
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # Randomly sample a bird and a background
        hair1 = self.hair[torch.randint(len(self.hair), (1,)).item()]
        celeb1 = self.celebs[torch.randint(len(self.celebs), (1,)).item()]

        hair2 = self.hair[torch.randint(len(self.hair), (1,)).item()]
        while hair1 == hair2:
            hair2 = self.hair[torch.randint(len(self.hair), (1,)).item()]
        
        celeb2 = self.celebs[torch.randint(len(self.celebs), (1,)).item()]
        while celeb1 == celeb2:
            celeb2 = self.celebs[torch.randint(len(self.celebs), (1,)).item()]

        

        # Create the sentence for negative samples
        sentence_neg_1 = f"A photo of a person with a {hair1} hair"
        sentence_neg_2 = f"A photo of a person with a {hair2} hair"
        
        # create the sentence for positive samples
        sentence_pos_1 = f"A photo of {celeb1} with a {hair1} hair."
        sentence_pos_2 = f"A photo of {celeb2} with a {hair1} hair."

        sentences = [sentence_neg_1, sentence_neg_2, sentence_pos_1, sentence_pos_2]

        return sentences
    

def get_SSL_dataset(args):
    # Define lists of birds and backgrounds
    land_birds = [
        "sparrow", "robin", "finch", "woodpecker", "canary", "hawk", "falcon", 
        "eagle", "owl", "crow", "pigeon", "dove", "cardinal", "jay", "wren", 
        "magpie", "partridge", "quail", "pheasant", "mockingbird"
    ]

    water_birds = [
        "duck", "swan", "pelican", "heron", "flamingo", "seagull", "penguin", 
        "albatross", "cormorant", "grebe", "loon", "kingfisher", "stork", "crane", 
        "ibis", "sandpiper", "tern", "plover", "puffin", "shearwater"
    ]

    birds = [land_birds , water_birds]

    water_backgrounds = ["lake", "sea", "ocean", "river", "pond", "stream", "marsh", "bay", "lagoon", "waterfall"]
    ground_backgrounds = ["forest", "field", "desert", "mountain", "garden", "savannah", "park", "backyard", "canyon", "cliff"]

    backgrounds = [water_backgrounds , ground_backgrounds]



    # List of male celebrities
    # List of male celebrities
    male_celebrities = [
        "Brad Pitt", "Leonardo DiCaprio", "Chris Hemsworth", "Robert Downey Jr.", "Dwayne Johnson",
        "Ryan Gosling", "Tom Holland",  
        "Will Smith", "Keanu Reeves", "Idris Elba", "Chris Evans", "Hugh Jackman",
        "Daniel Craig", "Henry Cavill", 
        "Jason Momoa", "Jake Gyllenhaal", "Tom Hardy", "Michael B. Jordan", "Mark Ruffalo",
        "Matthew McConaughey", "Ryan Reynolds", 
        "Timothée Chalamet", "Ben Affleck", "Paul Rudd", "Oscar Isaac", "Christian Bale",
        "Johnny Depp", "Eddie Redmayne"
    ]

    # List of female celebrities
    female_celebrities = [
        "Scarlett Johansson", "Angelina Jolie", "Jennifer Lawrence", "Margot Robbie", "Emma Watson",
        "Gal Gadot", "Zendaya",  
        "Anne Hathaway", "Saoirse Ronan", "Natalie Portman", "Emma Stone", "Priyanka Chopra",
        "Sandra Bullock", "Nicole Kidman", 
        "Charlize Theron", "Amy Adams", "Cate Blanchett", "Jessica Chastain", "Emily Blunt",
        "Lupita Nyong'o", "Meryl Streep", 
        "Rihanna", "Selena Gomez", "Taylor Swift", "Lady Gaga", "Beyoncé",
        "Ariana Grande", "Kylie Jenner"
    ]

    male_celebrities = ['man', 'male', 'mester']
    female_celebrities = ['woman', 'female', 'lady']

    celebs = [male_celebrities , female_celebrities]

    # # List of words describing dark hair
    # dark_hair = [
    #     "jet black", "raven", "midnight", "onyx", "charcoal",
    #     "ebony", "deep brown", "espresso", "mahogany", "sable",
    #     "chocolate", "ink", "coal", "obsidian", "smoky",
    #     "shadowy", "chestnut", "mocha", "brunette", "noir"
    # ]

    # # List of words describing blond hair
    # blond_hair = [
    #     "golden", "honey", "platinum", "sandy", "sun-kissed",
    #     "buttery", "ash blond", "straw", "champagne", "caramel",
    #     "pearl", "flaxen", "vanilla", "wheat", "light gold",
    #     "ivory", "fair", "sunlit", "lemon", "pale yellow"
    # ]

    dark_hair = ["dark", "black", "brunette"]
    blond_hair = ["blond", "blonde", "light"]


    hair = [dark_hair , blond_hair]


    if args.dataset == 'celeba':
        textset = Celeb_SSL_Dataset(celebs, hair, num_samples=args.num_samples)
        textloader = DataLoader(textset, batch_size=3, shuffle=True)
    elif args.dataset == 'waterbirds':
        textset = Bird_SSL_Dataset( birds, backgrounds, num_samples=args.num_samples, seed=args.seed)
        textloader = DataLoader(textset, batch_size=3, shuffle=True)
    
    print("data example")
    print(textset[0])
    return textloader
