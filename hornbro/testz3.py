from z3 import *

# Create the Z3 optimizer
opt = Optimize()

# Set a timeout of 100,000 milliseconds (100 seconds)
opt.set("timeout", 100000)

# Define Boolean variables
x = Bool('x')
y = Bool('y')
z = Bool('z')
nx = Bool('x')


# Add hard constraints (these must be satisfied)
opt.add(x == True)  # Example hard constraint

# Add soft constraints (these are clauses we want to satisfy as many as possible)
opt.add_soft(Or(Not(x), y))  # Clause 1
opt.add_soft(Or(Not(y), z))  # Clause 2
opt.add_soft(Or(Not(z), x))  # Clause 3
opt.add_soft(Or(x, y, z))    # Clause 4

# Maximize the number of satisfied soft constraints
print("Maximizing the number of satisfied clauses...")

# Check the solution with a timeout
result = opt.check()
opt.set("timeout", 1)
if result == sat:
    model = opt.model()
    print("Satisfiable with maximum satisfied clauses:")
    print(f"x = {model[x]}")
    print(f"y = {model[y]}")
    print(model.eval(Or(Not(x), y)==False))
    print(f"z = {model[z]}")
elif result == unknown:
    print("Timeout occurred, retrieving the best solution found so far:")
    model = opt.model()
    print(f"x = {model[x]}")
    
    print(model.eval(Or(Not(x), y)))
    print(f"y = {model[y]}")
    print(f"z = {model[z]}")
else:
    print("No solution found")
