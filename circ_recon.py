import random
from PIL import Image, ImageDraw, ImageFont

"""
Individual:
    num: determines how many circles to use to draw image 
    x: x position on canvas
    y: y position on canvas
    radius: radius for circular brush
    r: red value between 0-255
    g: green value between 0-255
    b: blue value between 0-255 
    alpha: transparency value between 0-255
"""

def create_initial_population(population_size, num_circles, width, height) -> list:
    """
    Creates the initial population.
    Each individual is a list of [num_circles] circle-genes.

    Args:
        population_size: determines size of population
        num_circles: number of circles used per drawing
        width: Canvas width, keep x/radius in bounds.
        height: Canvas height, keep x/radius in bounds.

    Returns: population list
    """

    population = []
    max_radius = max(1, min(width, height) // 10)
    for _ in range(population_size):
        individuals = []
        for _ in range(num_circles):
            individual = (random.randint(0, width - 1), #randomizing x value
                        random.randint(0, height - 1), #randomizing y value
                        random.randint(1, max_radius), #randomizing radius value
                        random.randint(0, 255), #randomizing r value
                        random.randint(0, 255), #randomizing g value
                        random.randint(0, 255), #randomizing b value
                        random.randint(20, 255) #randomizing alpha value, keeping it semi-transparent
                        )
            individuals.append(individual)
        population.append(individuals)
    
    return population

def render_individual(individuals, width, height):
    """
    Render an individual (list of circle genes) into an RGB image.

    Args:
        individuals: List of genes representing one drawing (each gene is (x, y, radius R, G, B, A)).
        width: Canvas width, keep x/radius in bounds.
        height: Canvas height, keep x/radius in bounds.

    Returns: image
    """
    layer = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer, "RGBA")
    for (x, y, radius, r, g, b, a) in individuals: #ensure sane bbox even if a stray value sneaks in
        x0, x1 = sorted((x - radius, x + radius))
        y0, y1 = sorted((y - radius, y + radius))
        x0 = clamp(x0, 0, width-1);  x1 = clamp(x1, 0, width-1)
        y0 = clamp(y0, 0, height-1); y1 = clamp(y1, 0, height-1)
        if x1 >= x0 and y1 >= y0:
            draw.ellipse((x0, y0, x1, y1), fill=(r, g, b, a))
    base = Image.new("RGBA", (width, height), (255, 255, 255, 255))
    return Image.alpha_composite(base, layer).convert("RGB")

def fitness_function(individuals, target_pixels, width, height) -> int:
    """
    Compute fitness as the negative sum of per-pixel color differences between the rendered individual and the target image (more negative = worse, closer to 0 = better).

    Args:
        individuals: List of genes representing one drawing (each gene is (x, y, radius R, G, B, A)).
        target_pixels: Flattened list of (R, G, B) pixels from the target image, used for fitness.
        width: Canvas width, keep x/radius in bounds.
        height: Canvas height, keep x/radius in bounds.

    Returns: fitness score (int)
    """
    drawn = render_individual(individuals, width, height)
    drawn_pixels = drawn.getdata()
    total = 0
    for dp, tp in zip(drawn_pixels, target_pixels):
        total += abs(dp[0]-tp[0]) + abs(dp[1]-tp[1]) + abs(dp[2]-tp[2])
    return -total #negative to 0 range

def selection(population, fitnesses, tournament_size) -> list:
    """
    Determines which individuals should reproduce to make next population set

    Args:
        population: list of individuals in population
        fitnesses: list of fitnesses corresponding to individual
        tournament_size: integer for how many individuals are put together for evaluation
    
    Returns: list of selected winners from tournament
    """

    selected = []
    for _ in range(len(population)):
        tournament = random.sample(list(zip(population, fitnesses)), tournament_size) #tournament will hold a list of pairs based on tournament_size - [(individual1, score1), (individual2, score2)...]
        winner = max(tournament, key=lambda x: x[1])[0] #compares the tournamnet with max by looking at score "x[1]". After finding, take the individual with [0] and store in winner
        selected.append(winner)
    return selected

def crossover(parent1, parent2):
    """
    Takes half of the parents genes and swaps between the two

    For each gene position, randomly choose which parent contributes to a child with 0.5 probability.
    
    Args:
        parent1: a winner from the competition pool
        parent2: other winner from the competition pool
    
    Returns: tuple of list c1 and c2, the new genes
    """
    c1, c2 = [], []
    for g1, g2 in zip(parent1, parent2):
        if random.random() < 0.5:
            c1.append(g1); c2.append(g2)
        else:
            c1.append(g2); c2.append(g1)
    return c1, c2

def clamp(v, lo, hi):
    """
    Clamp a value in a closed interval, lo - hi.

    Args:
        v: The value to be clamped.
        lo: Lower bound of range (inclusive).
        hi: Upper bound of range (inclusive).

    Returns:
        'v' if it lies between 'lo' and 'hi', otherwise 'lo' or 'hi' depending on which bound it exceeds.
    """
    return max(lo, min(hi, v))

def mutate_gene(g, width, height):
    """
    Mutate a single circle gene to introduce diversity.
    
    A gene is a 7-tuple: (x, y, radius, R, G, B, A).
    
    With 0.8 probabiliy (the randomness threshold), the mutation makes small changes to the current gene parameters. With 0.2 probability, the gene is completely re-randomized

    Args:
        g: The current gene, as a 7-tuple (x, y, radius, R, G, B, A).
        width: Canvas width, keep x/radius in bounds.
        height: Canvas height, keep x/radius in bounds.

    Returns: mutated gene in the same (x, y, radius, R, G, B, A) format.
    """
    x, y, r, R, G, B, A = g
    max_r = max(2, min(width, height)//10)
    if random.random() < 0.8:
        x = clamp(int(round(random.gauss(x, width*0.05))),  0, width-1)
        y = clamp(int(round(random.gauss(y, height*0.05))), 0, height-1)
        r = clamp(int(round(random.gauss(r, 2))), 1, max_r) #radius >= 1
        R = clamp(int(round(random.gauss(R, 20))), 0, 255)
        G = clamp(int(round(random.gauss(G, 20))), 0, 255)
        B = clamp(int(round(random.gauss(B, 20))), 0, 255)
        A = clamp(int(round(random.gauss(A, 15))), 10, 255) #keep some transparency
        return (x, y, r, R, G, B, A)
    else:
        return (
            random.randint(0, width-1),
            random.randint(0, height-1),
            random.randint(2, max_r),
            random.randint(0, 255),
            random.randint(0, 255),
            random.randint(0, 255),
            random.randint(20, 180),
        )

def mutation(individual, mutation_rate, width, height):
    """
    Apply mutation to every gene in an individual with some probability.

    Each individual is a list of circle genes -- each gene, with probability
    'mutation_rate' we call 'mutate_gene' to slightly modify or re-randomize it. Otherwise, the gene is copied unchanged.

    Args:
        individual: List of genes representing one drawing (each gene is (x, y, radius R, G, B, A)).
        mutation_rate: Probability in 0-1 of mutating each gene.
        width: Canvas width, keep x/radius in bounds.
        height: Canvas height, keep x/radius in bounds.   
    
    Returns: A list of new individual with some genes mutated.
    """

    return [mutate_gene(g, width, height) if random.random() < mutation_rate else g
            for g in individual]


def genetic_algorithm(population_size, num_circles, generations, mutation_rate, target_pixels, width, height) -> list:
    """
    Run the full genetic algorithm to approximate a target image using circles

    *Algorithm:*

    - Initializes a random population of individuals
    - Repeats the following across 'generations':
        - Evaluates fitness of each individual against 'target_pixels'.
        - Tracks the best individual seen so far.
        - Selects parents via tournament selection.
        - Keeps a small elite set unchanged (elitism).
        - Builds the next generation via crossover + mutation.

    Args:
        population_size: Number of individuals per generation.
        num_circles: Number of circle genes in each individual.
        generations: Number of generations to evolve for.
        mutation_rate: Probability in 0-1 of mutating each gene.
        target_pixels: Flattened list of (R, G, B) pixels from the target image, used for fitness.
        width: Canvas width, keep x/radius in bounds.
        height: Canvas height, keep x/radius in bounds.

    Returns: single best individual found over all generations
    """

    population = create_initial_population(population_size, num_circles, width, height)
    best_so_far, best_fit_so_far = None, float("-inf")
    ELITE_K = max(1, population_size // 20)

    for gen in range(generations):
        fitnesses = [fitness_function(ind, target_pixels, width, height) for ind in population]

        #track
        gi = max(range(population_size), key=lambda i: fitnesses[i])
        if fitnesses[gi] > best_fit_so_far:
            best_fit_so_far, best_so_far = fitnesses[gi], population[gi]

        if gen % 10 == 0:
            print(f"Gen {gen:4d} | best(gen)={fitnesses[gi]} | best(so far)={best_fit_so_far}")

        #selection
        selected = selection(population, fitnesses, tournament_size=5)

        #elitism
        elite_idxs = sorted(range(population_size), key=lambda i: fitnesses[i], reverse=True)[:ELITE_K]
        next_population = [population[i] for i in elite_idxs]

        random.shuffle(selected)
        while len(next_population) < population_size:
            p1 = selected[len(next_population) % population_size]
            p2 = selected[(len(next_population)+1) % population_size]
            c1, c2 = crossover(p1, p2)
            next_population.append(mutation(c1, mutation_rate, width, height))
            if len(next_population) < population_size:
                next_population.append(mutation(c2, mutation_rate, width, height))

        population = next_population

    print("Optimization finished!")
    return best_so_far

def generate_circles(path, population_size, num_circles, generations, mutation_rate):
    """
    Takes in a path for the image to recreate using circles. Function also sets

    Args:
        path: location at which the image is stored (usually placed in "images" directory)
        population_size: Number of individuals per generation.
        num_circles: Number of circle genes in each individual.
        generations: Number of generations to evolve for.
        mutation_rate: Probability in 0-1 of mutating each gene.
    
    Returns: A list of new individual with some genes mutated.
    """
    target_image = Image.open(path).convert("RGB")
    img_target_pixels = list(target_image.getdata())
    img_width, img_height = target_image.size

    best_solution = genetic_algorithm(population_size, num_circles, generations, mutation_rate, img_target_pixels, img_width, img_height)

    print(f"Best solution: {best_solution}")

    final_img = render_individual(best_solution, img_width, img_height)
    final_img.show()
    final_img.save("Circle_google_100.png")

if __name__ == "__main__":
    generate_circles(r"InputImages\googlelogoresized.png", population_size=100, num_circles=500, generations=100, mutation_rate=0.075)