from tqdm import tqdm
from pathlib import Path as P

labels = P('./labels/')
new_labels = P('./new_labels/')


def main():
    new_labels.mkdir(exist_ok=True)
    
    for split in labels.iterdir():
        if split.name.endswith('.cache'): 
            continue

        (new_labels/split.name).mkdir(exist_ok=True)
        
        for f in tqdm(list(split.iterdir()), desc=f"Running on {split.name} split..."):     
            with open(f, 'r') as file:
                lines = file.readlines()
            
            new_lines = []
            for line in lines:
                new_line = ""
                label = line[0]
                
                if label == "5": # glove
                    new_line = "0" + line[1:]
                    
                elif label == "7": # no_glove (hand)
                    new_line = "1" + line[1:]
                
                if new_line:
                    new_lines.append(new_line)        
            
            new_file = new_labels/split.name/f.name
            with open(new_file, 'w') as file:
                file.writelines(new_lines)


if __name__ == "__main__":
    main()