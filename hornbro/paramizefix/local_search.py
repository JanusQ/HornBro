# Genetic Algorithm for optimizing circuit structure
def genetic_algorithm_optimize(circuit, correct_output_states, input_states,population_size=10, generations=20, mutation_rate=0.1):
    def create_population(circuit, size):
        return [generate_bugged_circuit(circuit, n_errors= len(circuit)// 2) for _ in range(size)]

    def evaluate_fitness(circuit, correct_output_states):
        total_distance = 0
        output_states = apply_circuit(circuit, input_states)
        for output_state, correct_output in zip(output_states, correct_output_states):
            total_distance += calculate_distance(output_state, correct_output)
        return total_distance


    def select_parents(population, fitnesses):
        fitness_sum = sum(fitnesses)
        probs = [f / fitness_sum for f in fitnesses]
        parents_indices = np.random.choice(range(len(population)), size=2, p=probs)
        return population[parents_indices[0]], population[parents_indices[1]]

    def crossover(parent1, parent2):
        crossover_point = random.randint(0, len(parent1.data) - 1)
        child1_data = parent1.data[:crossover_point] + parent2.data[crossover_point:]
        child2_data = parent2.data[:crossover_point] + parent1.data[crossover_point:]
        child1 = QuantumCircuit(*parent1.qregs)
        child2 = QuantumCircuit(*parent2.qregs)
        child1.data = child1_data
        child2.data = child2_data
        return child1, child2

    def mutate(circuit, mutation_rate):
        if random.random() < mutation_rate:
            circuit = generate_bugged_circuit(circuit, n_errors=len(circuit)//10)
        return circuit

    population = create_population(circuit, population_size)
    best_circuit = None
    best_fitness = float('inf')

    for generation in range(generations):
        fitnesses = [evaluate_fitness(individual, correct_output_states) for individual in population]
        for i, fitness in enumerate(fitnesses):
            if fitness < best_fitness:
                best_fitness = fitness
                best_circuit = population[i]

        new_population = []
        for _ in range(population_size // 2):
            parent1, parent2 = select_parents(population, fitnesses)
            child1, child2 = crossover(parent1, parent2)
            child1 = mutate(child1, mutation_rate)
            child2 = mutate(child2, mutation_rate)
            new_population.extend([child1, child2])
        
        population = new_population

    return best_circuit
