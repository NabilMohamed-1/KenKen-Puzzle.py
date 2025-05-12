import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import random
import time
from typing import List, Tuple, Dict, Set
from threading import Thread

class KenKenPuzzle:
    def __init__(self, size: int):
        self.size = size
        self.grid = np.zeros((size, size), dtype=int)
        self.cages = []
        self.solution = None
        self.metrics = {
            'time': 0,
            'success': False
        }

    def generate_puzzle(self):
        self.solution = self._generate_solution()
        self._generate_cages()

    def _generate_solution(self):
        solution = np.zeros((self.size, self.size), dtype=int)
        for i in range(self.size):
            nums = list(range(1, self.size + 1))
            random.shuffle(nums)
            solution[i][i] = nums[0]

        def is_valid(num, pos):
            for x in range(self.size):
                if solution[pos[0]][x] == num and pos[1] != x:
                    return False
            for x in range(self.size):
                if solution[x][pos[1]] == num and pos[0] != x:
                    return False
            return True

        def solve():
            for i in range(self.size):
                for j in range(self.size):
                    if solution[i][j] == 0:
                        for num in range(1, self.size + 1):
                            if is_valid(num, (i, j)):
                                solution[i][j] = num
                                if solve():
                                    return True
                                solution[i][j] = 0
                        return False
            return True

        solve()
        return solution

    def _generate_cages(self):
        self.cages = []
        used_cells = set()
        while len(used_cells) < self.size * self.size:
            available_cells = [(i, j) for i in range(self.size) for j in range(self.size) if (i, j) not in used_cells]
            start_cell = random.choice(available_cells)
            cage_cells = [start_cell]
            used_cells.add(start_cell)
            cage_size = random.randint(1, min(4, self.size))
            while len(cage_cells) < cage_size:
                current = cage_cells[-1]
                adjacents = []
                for di, dj in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                    ni, nj = current[0] + di, current[1] + dj
                    if (0 <= ni < self.size and 0 <= nj < self.size and (ni, nj) not in used_cells):
                        adjacents.append((ni, nj))
                if not adjacents:
                    break
                next_cell = random.choice(adjacents)
                cage_cells.append(next_cell)
                used_cells.add(next_cell)

            values = [self.solution[i][j] for i, j in cage_cells]
            if len(cage_cells) == 1:
                target = values[0]
                operation = ""
            else:
                if random.random() < 0.4:
                    operation = "+"
                    target = sum(values)
                elif random.random() < 0.7:
                    operation = "*"
                    target = np.prod(values)
                elif len(cage_cells) == 2:
                    if random.random() < 0.5:
                        operation = "-"
                        target = abs(values[0] - values[1])
                    else:
                        operation = "/"
                        target = max(values[0] / values[1], values[1] / values[0])
                else:
                    operation = "+"
                    target = sum(values)

            self.cages.append((cage_cells, operation, target))

    def is_valid(self) -> bool:
        for i in range(self.size):
            row = self.grid[i, :]
            col = self.grid[:, i]
            if len(set(row)) != len(row) or len(set(col)) != len(col):
                return False
        for cells, operation, target in self.cages:
            values = [self.grid[r, c] for r, c in cells]
            if 0 in values:
                continue
            if operation == '+':
                if sum(values) != target:
                    return False
            elif operation == '-':
                if abs(values[0] - values[1]) != target:
                    return False
            elif operation == '*':
                if np.prod(values) != target:
                    return False
            elif operation == '/':
                if max(values[0] / values[1], values[1] / values[0]) != target:
                    return False
        return True

class BacktrackingSolver:
    def __init__(self, puzzle: KenKenPuzzle, update_callback=None):
        self.puzzle = puzzle
        self.update_callback = update_callback
        self.last_update_time = time.time()

    def solve(self) -> bool:
        start_time = time.time()
        self.puzzle.grid = np.zeros((self.puzzle.size, self.puzzle.size), dtype=int)
        result = self._solve()
        self.puzzle.metrics = {
            'time': time.time() - start_time,
            'success': result
        }
        return result

    def _solve(self) -> bool:
        current_time = time.time()
        if current_time - self.last_update_time > 0.1:
            if self.update_callback:
                self.update_callback(self.puzzle.grid)
            self.last_update_time = current_time

        empty = self._find_empty()
        if not empty:
            return True

        row, col = empty
        numbers = list(range(1, self.puzzle.size + 1))
        random.shuffle(numbers)
        
        for num in numbers:
            if self._is_valid_move(row, col, num):
                self.puzzle.grid[row][col] = num
                if self._solve():
                    return True
                self.puzzle.grid[row][col] = 0

        return False

    def _is_valid_move(self, row: int, col: int, num: int) -> bool:
        for x in range(self.puzzle.size):
            if self.puzzle.grid[row][x] == num and x != col:
                return False
        for x in range(self.puzzle.size):
            if self.puzzle.grid[x][col] == num and x != row:
                return False
        
        for cells, operation, target in self.puzzle.cages:
            if (row, col) in cells:
                values = [self.puzzle.grid[r][c] for r, c in cells if (r, c) != (row, col)]
                values.append(num)
                if 0 in values:
                    continue
                    
                if operation == '+':
                    if sum(values) != target:
                        return False
                elif operation == '-':
                    if len(values) == 2 and abs(values[0] - values[1]) != target:
                        return False
                elif operation == '*':
                    if np.prod(values) != target:
                        return False
                elif operation == '/':
                    if len(values) == 2 and max(values[0] / values[1], values[1] / values[0]) != target:
                        return False
        return True

    def _find_empty(self) -> Tuple[int, int]:
        min_possibilities = float('inf')
        best_cell = None
        
        for i in range(self.puzzle.size):
            for j in range(self.puzzle.size):
                if self.puzzle.grid[i][j] == 0:
                    possibilities = 0
                    for num in range(1, self.puzzle.size + 1):
                        if self._is_valid_move(i, j, num):
                            possibilities += 1
                    if possibilities < min_possibilities:
                        min_possibilities = possibilities
                        best_cell = (i, j)
        
        return best_cell

class GeneticSolver:
    def __init__(self, puzzle: KenKenPuzzle, population_size=100, generations=1000, update_callback=None):
        self.puzzle = puzzle
        self.population_size = population_size
        self.generations = generations
        self.update_callback = update_callback
        self.last_update_time = time.time()

    def solve(self) -> bool:
        start_time = time.time()
        self.puzzle.grid = np.zeros((self.puzzle.size, self.puzzle.size), dtype=int)
        population = self._initialize_population()
        best_fitness = 0
        best_solution = None
        
        for generation in range(self.generations):
            current_time = time.time()
            if current_time - self.last_update_time > 0.1:
                if self.update_callback:
                    self.update_callback(best_solution if best_solution is not None else self.puzzle.grid)
                self.last_update_time = current_time

            fitness_scores = [self._calculate_fitness(ind) for ind in population]
            current_best = max(fitness_scores)
            current_best_idx = fitness_scores.index(current_best)
            
            if current_best > best_fitness:
                best_fitness = current_best
                best_solution = population[current_best_idx]
                if self.update_callback:
                    self.update_callback(best_solution)

            if best_fitness == 1.0:
                self.puzzle.grid = best_solution
                break

            new_population = []
            elite_size = 5
            elite_indices = sorted(range(len(fitness_scores)), 
                                key=lambda i: fitness_scores[i], 
                                reverse=True)[:elite_size]
            new_population.extend([population[i] for i in elite_indices])

            while len(new_population) < self.population_size:
                parent1 = self._select_parent(population, fitness_scores)
                parent2 = self._select_parent(population, fitness_scores)
                child = self._crossover(parent1, parent2)
                child = self._mutate(child)
                child = self._improve_solution(child)
                new_population.append(child)

            population = new_population

        self.puzzle.metrics = {
            'time': time.time() - start_time,
            'success': best_fitness == 1.0
        }
        
        if best_fitness == 1.0:
            self.puzzle.grid = best_solution
            return True
        return False

    def _initialize_population(self) -> List[np.ndarray]:
        population = []
        for _ in range(self.population_size):
            individual = np.zeros((self.puzzle.size, self.puzzle.size), dtype=int)
            for i in range(self.puzzle.size):
                for j in range(self.puzzle.size):
                    valid_numbers = self._get_valid_numbers(i, j, individual)
                    if valid_numbers:
                        individual[i][j] = random.choice(valid_numbers)
                    else:
                        individual[i][j] = random.randint(1, self.puzzle.size)
            population.append(individual)
        return population

    def _get_valid_numbers(self, row: int, col: int, grid: np.ndarray) -> List[int]:
        valid_numbers = []
        for num in range(1, self.puzzle.size + 1):
            if self._is_valid_move(row, col, num, grid):
                valid_numbers.append(num)
        return valid_numbers

    def _is_valid_move(self, row: int, col: int, num: int, grid: np.ndarray) -> bool:
        for x in range(self.puzzle.size):
            if grid[row][x] == num and x != col:
                return False
        for x in range(self.puzzle.size):
            if grid[x][col] == num and x != row:
                return False
        
        for cells, operation, target in self.puzzle.cages:
            if (row, col) in cells:
                values = [grid[r][c] for r, c in cells if (r, c) != (row, col)]
                values.append(num)
                if 0 in values:
                    continue
                    
                if operation == '+':
                    if sum(values) != target:
                        return False
                elif operation == '-':
                    if len(values) == 2 and abs(values[0] - values[1]) != target:
                        return False
                elif operation == '*':
                    if np.prod(values) != target:
                        return False
                elif operation == '/':
                    if len(values) == 2 and max(values[0] / values[1], values[1] / values[0]) != target:
                        return False
        return True

    def _calculate_fitness(self, grid: np.ndarray) -> float:
        score = 0
        total_constraints = self.puzzle.size * 2 + len(self.puzzle.cages)
        
        for i in range(self.puzzle.size):
            if len(set(grid[i])) == self.puzzle.size:
                score += 1
            if len(set(grid[:, i])) == self.puzzle.size:
                score += 1
        
        for cells, operation, target in self.puzzle.cages:
            values = [grid[i][j] for i, j in cells]
            if self._check_cage_constraint(values, operation, target):
                score += 1
        
        return score / total_constraints

    def _check_cage_constraint(self, values: List[int], operation: str, target: int) -> bool:
        if operation == '+':
            return sum(values) == target
        elif operation == '-':
            return len(values) == 2 and abs(values[0] - values[1]) == target
        elif operation == '*':
            return np.prod(values) == target
        elif operation == '/':
            return len(values) == 2 and max(values[0] / values[1], values[1] / values[0]) == target
        return True

    def _select_parent(self, population: List[np.ndarray], fitness_scores: List[float]) -> np.ndarray:
        tournament_size = 3
        tournament = random.sample(list(enumerate(fitness_scores)), tournament_size)
        winner_idx = max(tournament, key=lambda x: x[1])[0]
        return population[winner_idx]

    def _crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> np.ndarray:
        child = np.zeros((self.puzzle.size, self.puzzle.size), dtype=int)
        for i in range(self.puzzle.size):
            if random.random() < 0.5:
                child[i] = parent1[i]
            else:
                child[i] = parent2[i]
        return child

    def _mutate(self, grid: np.ndarray) -> np.ndarray:
        mutated = grid.copy()
        if random.random() < 0.3:
            row = random.randint(0, self.puzzle.size - 1)
            col = random.randint(0, self.puzzle.size - 1)
            valid_numbers = self._get_valid_numbers(row, col, mutated)
            if valid_numbers:
                mutated[row][col] = random.choice(valid_numbers)
        return mutated

    def _improve_solution(self, grid: np.ndarray) -> np.ndarray:
        improved = grid.copy()
        for i in range(self.puzzle.size):
            for j in range(self.puzzle.size):
                valid_numbers = self._get_valid_numbers(i, j, improved)
                if valid_numbers:
                    improved[i][j] = random.choice(valid_numbers)
        return improved

class KenKenCell(tk.Frame):
    def __init__(self, master, size=80, **kwargs):
        super().__init__(master, width=size, height=size, **kwargs)
        self.size = size
        self.pack_propagate(False)
        
        self.canvas = tk.Canvas(self, width=size, height=size, 
                              highlightthickness=0, bg='white')
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        self.value = tk.StringVar()
        self.entry = ttk.Entry(self.canvas, width=2, justify="center",
                             textvariable=self.value, font=('Arial', 24, 'bold'))
        self.canvas.create_window(size/2, size/2, window=self.entry)
        
        self.borders = {"top": 1, "right": 1, "bottom": 1, "left": 1}
        self.operation_text = ""
        
    def set_borders(self, borders):
        self.borders = borders
        self.redraw()
        
    def set_operation(self, text):
        self.operation_text = text
        self.redraw()
        
    def redraw(self):
        self.canvas.delete("all")
        
        for side, width in self.borders.items():
            if width > 0:
                if side == "top":
                    self.canvas.create_line(0, 0, self.size, 0, 
                                         width=width, fill='black')
                elif side == "right":
                    self.canvas.create_line(self.size-width/2, 0, self.size-width/2, self.size, 
                                         width=width, fill='black')
                elif side == "bottom":
                    self.canvas.create_line(0, self.size-width/2, self.size, self.size-width/2, 
                                         width=width, fill='black')
                elif side == "left":
                    self.canvas.create_line(0, 0, 0, self.size, 
                                         width=width, fill='black')
        
        if self.operation_text:
            self.canvas.create_rectangle(0, 0, self.size/2, self.size/2, 
                                      fill='white', outline='white')
            self.canvas.create_text(self.size/4, self.size/4, 
                                  text=self.operation_text,
                                  font=('Arial', 16, 'bold'),
                                  fill='red')
        
        self.canvas.create_window(self.size/2, self.size/2, window=self.entry)

class KenKenGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("KenKen Puzzle Solver")
        
        self.container = ttk.Frame(self.root)
        self.container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.left_panel = ttk.Frame(self.container)
        self.left_panel.pack(side=tk.LEFT, padx=(0, 10))
        
        self.controls = ttk.LabelFrame(self.left_panel, text="Controls", padding=5)
        self.controls.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(self.controls, text="Algorithm:").pack(side=tk.LEFT)
        self.algorithm_var = tk.StringVar(value="Backtracking")
        self.algorithm_combo = ttk.Combobox(self.controls, 
                                          textvariable=self.algorithm_var,
                                          values=["Backtracking", "Genetic"],
                                          width=12)
        self.algorithm_combo.pack(side=tk.LEFT, padx=5)
        
        ttk.Button(self.controls, text="Generate Puzzle", 
                  command=self.generate_new_puzzle).pack(side=tk.LEFT, padx=5)
        ttk.Button(self.controls, text="Solve", 
                  command=self.solve_puzzle).pack(side=tk.LEFT, padx=5)
        ttk.Button(self.controls, text="Clear", 
                  command=self.clear_puzzle).pack(side=tk.LEFT, padx=5)
        
        self.grid_frame = ttk.Frame(self.left_panel)
        self.grid_frame.pack()
        
        self.right_panel = ttk.Frame(self.container)
        self.right_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.right_panel = ttk.Frame(self.container)
        self.right_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        self.metrics_frame = ttk.LabelFrame(self.right_panel, text="Performance Metrics", padding=5)
        self.metrics_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.time_var = tk.StringVar(value="Time: 0.00s")
        self.status_var = tk.StringVar(value="Status: Ready")
        
        ttk.Label(self.metrics_frame, textvariable=self.time_var).pack(anchor=tk.W)
        ttk.Label(self.metrics_frame, textvariable=self.status_var).pack(anchor=tk.W)
        
        self.puzzle = KenKenPuzzle(6)
        self.puzzle.generate_puzzle() 
        
        self.cells = []
        self.create_grid()
        self.setup_puzzle()
        
        self.original_solution = self.puzzle.solution.copy()
    
    def generate_new_puzzle(self):
        self.puzzle = KenKenPuzzle(6)
        self.puzzle.generate_puzzle()
        
        self.original_solution = self.puzzle.solution.copy()
        
        self.clear_puzzle()
        
        for i in range(6):
            for j in range(6):
                self.cells[i][j].value.set(str(self.puzzle.solution[i][j]))
        
        self.setup_puzzle()
        
        self.status_var.set("Status: New puzzle generated")
    
    def create_grid(self):
        for i in range(6):
            row = []
            for j in range(6):
                cell = KenKenCell(self.grid_frame)
                cell.grid(row=i, column=j)
                row.append(cell)
            self.cells.append(row)
    
    def setup_puzzle(self):
        for cells, operation, target in self.puzzle.cages:
            first_cell = cells[0]
            self.cells[first_cell[0]][first_cell[1]].set_operation(f"{target}{operation}")
            
            for i, j in cells:
                borders = {"top": 1, "right": 1, "bottom": 1, "left": 1}
                
                for di, dj, border in [(0, -1, "left"), (0, 1, "right"),
                                     (-1, 0, "top"), (1, 0, "bottom")]:
                    ni, nj = i + di, j + dj
                    if (ni, nj) not in cells:
                        borders[border] = 4
                
                self.cells[i][j].set_borders(borders)
    
    def update_display(self, grid):
        for i in range(6):
            for j in range(6):
                self.cells[i][j].value.set(str(grid[i][j]) if grid[i][j] != 0 else "")
        
        self.root.update()
        
    def update_metrics(self, solver_metrics):
        self.time_var.set(f"Time: {solver_metrics['time']:.2f}s")
        self.status_var.set(f"Status: {'Success' if solver_metrics['success'] else 'Failed'}")
    
    def solve_puzzle(self):
        self.time_var.set("Time: 0.00s")
        self.status_var.set("Status: Solving...")
        self.root.update()
        
        for i in range(6):
            for j in range(6):
                value = self.cells[i][j].value.get()
                self.puzzle.grid[i][j] = int(value) if value else 0
        
        if self.algorithm_var.get() == "Backtracking":
            solver = BacktrackingSolver(self.puzzle, self.update_display)
        else:
            solver = GeneticSolver(self.puzzle, update_callback=self.update_display)
        
        success = solver.solve()
        
        if success:
            for i in range(6):
                for j in range(6):
                    self.cells[i][j].value.set(str(self.puzzle.grid[i][j]))
            
            self.update_metrics(self.puzzle.metrics)
            
            if np.array_equal(self.puzzle.grid, self.original_solution):
                self.status_var.set("Status: Success (matches original solution)")
            else:
                self.status_var.set("Status: Success (different from original solution)")
        else:
            self.status_var.set("Status: No solution found")
    
    def clear_puzzle(self):
        for row in self.cells:
            for cell in row:
                cell.value.set("")
        self.status_var.set("Status: Ready")
        self.time_var.set("Time: 0.00s")
    
    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    app = KenKenGUI()
    app.run()
        
