# Circle-Based Image Approximation with Genetic Algorithms

### Brief Overview:

This project uses a genetic algorithm (GA) to approximate a target image using many semi-transparent circles. Each candidate solution (an individual) is defined as a list of circle "genes" where every gene encodes the circle’s position, radius, color, and alpha transparency.

The core image operations are handled with the **Pillow** library. The **Random** library is also utilized for gene mutation and randomization. For each individual, the script draws all of its circles onto a blank canvas and compares the resulting image to the target image. The GA then iteratively evolves the population of individuals to minimize the pixel-wise color difference from the target.

### Key components include:

_create_initial_population_ – randomly generates an initial population of circle-based images.

_render_individual_ – draws all circles for a single individual and returns the rendered image.

_fitness_function_ – evaluates how close a rendered image is to the target (lower error = higher fitness).

_selection, crossover, mutation, mutate_gene_ – standard GA operators adapted for image approximation.

_genetic_algorithm_ – runs the full GA loop across multiple generations.

_generate_circles_ – convenience function that loads an input image, runs the GA, and saves the final result.

### Use Case:

Ensure Python and the required libraries are installed (should be Pillow and Random). 

Place the image you want to approximate in a directory (i.e., InputImages/).

In the generate_circles specify the path to that image and tune parameters: 

(i.e. generate_circles(r"InputImages\IMAGE_NAME.png",
                 population_size=100,
                 num_circles=500,
                 generations=100,
                 mutation_rate=0.075))

### Tuning Parameters:

_population_size:_ Larger populations explore more candidates each generation but take longer to evaluate.

_num_circles:_ More circles can capture finer details but increase rendering and evaluation cost.

_generations:_ More generations usually improve the approximation but increase runtime and processing.

_mutation_rate:_ Controls how often genes are altered; too low = very little change, will not deviate -- too high = too random and unpredictable.

### Visual Samples:

![google_logo](/InputImages/googlelogoresized.png)

_^^^ Example Input Image_

![Circle_google_100](/Output/Circle_google_100.png)

_^^^ Figure 1. Output of script after 100 generations for a Google logo

![Circle_google_1000](/Output/Circle_google_1000.png)

_^^^ Figure 2. Output of script after 1000 generations for a Google logo with a random seed

![Circle_google_1000b](/Output/Circle_google_1000b.png)

_^^^ Figure 3. Output of script after 1000 generations for a Google logo with a different random seed

![microsoft_logo](/InputImages/microsoftlogoresized.png)

_^^^ Example Input Image 2_

![Circle_microsoft_100](/Output/Circle_microsoft_100.png)

_^^^ Figure 1. Output of script after 100 generations for a Microsoft logo

![Circle_microsoft_1000](/Output/Circle_microsoft_1000.png)

_^^^ Figure 1. Output of script after 1000 generations for a Microsoft logo

### Resources:

Pillow documentation: https://pypi.org/project/pillow/

Introductory genetic algorithm tutorial in Python: https://www.datacamp.com/tutorial/genetic-algorithm-python
