from scipy.io import loadmat

# Load the .mat file
mat_data = loadmat('allPlanesVariables27-Feb-2021.mat')

# Remove metadata entries like '__header__', '__version__', '__globals__'
cleaned_data = {k: v for k, v in mat_data.items() if not k.startswith('__')}

# Write to a text file
with open('output.txt', 'w') as f:
    for key, value in cleaned_data.items():
        f.write(f"{key}:\n{value}\n\n")
