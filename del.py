import json

file_path = r'data\OVERT-full\violence_full.json'  # Replace with your file path
with open(file_path, 'r') as file:
    data = json.load(file)  
print(len(data))
new_data = []
for prompt_dict in data:
    if len(prompt_dict['image_prompts']) > 0:
        new_data.append(prompt_dict)
print(len(new_data))
# Save the modified data back to the file if needed
with open(file_path, 'w') as file:    
    json.dump(new_data, file, indent=4)  # Save the modified data back to the file