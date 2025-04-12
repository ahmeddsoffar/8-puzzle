import tkinter as tk
from tkinter import messagebox, ttk, font
import time
from collections import deque
import heapq
import math
import random
import threading

# Constants for styling
BACKGROUND_COLOR = "#f0f0f0"
BUTTON_COLOR = "#e0e0e0"
EMPTY_COLOR = "#f8f8f8"
TITLE_COLOR = "#2c3e50"
ACCENT_COLOR = "#3498db"
BUTTON_ACTIVE_COLOR = "#2980b9"
TEXT_COLOR = "#2c3e50"

goal_state = "012345678"

def find_zero_position(state):
    return state.index('0')

def get_neighbors(state):
    neighbors = []
    zero_index = find_zero_position(state)
    row, col = divmod(zero_index, 3)
    moves = {'Right': (0, 1), 'Up': (-1, 0), 'Left': (0, -1), 'Down': (1, 0)}
 
    for move, (dr, dc) in moves.items():
        new_row, new_col = row + dr, col + dc
        if 0 <= new_row < 3 and 0 <= new_col < 3:
            new_index = new_row * 3 + new_col
            new_state = list(state)
            new_state[zero_index], new_state[new_index] = new_state[new_index], new_state[zero_index]
            neighbors.append(("".join(new_state), move)) 
    return neighbors

def bfs(initial_state, report_progress=None):
    queue = deque([(initial_state, [])])
    visited = set()
    visited.add(initial_state)
    nodes_expanded = 0
    start_time = time.time()
    
    while queue:
        state, path = queue.popleft()
        nodes_expanded += 1
        
        if report_progress:
            report_progress(nodes_expanded, len(path))
        
        if state == goal_state:
            end_time = time.time()
            result = {
                "algorithm": "BFS",
                "path": path,
                "cost": len(path),
                "nodes_expanded": nodes_expanded,
                "search_depth": len(path),
                "running_time": end_time - start_time
            }
            return result
        
        for neighbor, move in get_neighbors(state):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, path + [move]))
    
    return None

def dfs(initial_state, report_progress=None):
    stack = [(initial_state, [])]
    visited = set()
    nodes_expanded = 0
    start_time = time.time()
    max_depth = 0
    
    while stack:
        state, path = stack.pop()
        
        if state in visited:
            continue
        
        visited.add(state)
        nodes_expanded += 1
        max_depth = max(max_depth, len(path))
        
        if report_progress:
            report_progress(nodes_expanded, max_depth)
        
        if state == goal_state:
            end_time = time.time()
            result = {
                "algorithm": "DFS",
                "path": path,
                "cost": len(path),
                "nodes_expanded": nodes_expanded,
                "search_depth": max_depth,
                "running_time": end_time - start_time
            }
            return result
        
        for neighbor, move in get_neighbors(state):
            if neighbor not in visited:
                stack.append((neighbor, path + [move]))
    
    return None

def dfs_limited(initial_state, max_depth, report_progress=None, expanded=0):
    stack = [(initial_state, [], 0)]
    visited = set()
    nodes_expanded = 0
    current_max_depth = 0

    while stack:
        state, path, depth = stack.pop()

        if state in visited:
            continue

        visited.add(state)
        nodes_expanded += 1
        current_max_depth = max(current_max_depth, depth)
        
        if report_progress:
            report_progress(expanded + nodes_expanded, current_max_depth)

        if state == goal_state:
            return path, nodes_expanded, current_max_depth

        if depth < max_depth:
            for neighbor, move in get_neighbors(state):
                if neighbor not in visited:
                    stack.append((neighbor, path + [move], depth + 1))

    return None, nodes_expanded, current_max_depth

def iterative_dfs(initial_state, max_depth=20, report_progress=None):
    start_time = time.time()
    total_nodes_expanded = 0
    max_search_depth = 0

    for depth in range(max_depth + 1):
        result, nodes_expanded, current_max_depth = dfs_limited(initial_state, depth, report_progress, total_nodes_expanded)
        total_nodes_expanded += nodes_expanded
        max_search_depth = max(max_search_depth, current_max_depth)
        
        if result is not None:
            end_time = time.time()
            result_dict = {
                "algorithm": f"IDDFS (max depth: {depth})",
                "path": result,
                "cost": len(result),
                "nodes_expanded": total_nodes_expanded,
                "search_depth": max_search_depth,
                "running_time": end_time - start_time
            }
            return result_dict

    return None

def heuristic_distance(state, heuristic='manhattan'):
    distance = 0
    for i, tile in enumerate(state):
        if tile != '0':
            goal_index = goal_state.index(tile)
            x1, y1 = divmod(i, 3)
            x2, y2 = divmod(goal_index, 3)
            
            if heuristic == 'manhattan':
                distance += abs(x1 - x2) + abs(y1 - y2)
            elif heuristic == 'euclidean':
                distance += math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

    return distance

def a_star(initial_state, heuristic='manhattan', report_progress=None):
    start_time = time.time()
    
    priority_queue = [] 
    heapq.heappush(priority_queue, (heuristic_distance(initial_state, heuristic), 0, initial_state, []))
    
    visited = set()
    nodes_expanded = 0
    max_depth = 0

    while priority_queue:
        _, cost, state, path = heapq.heappop(priority_queue)
        
        if state in visited:
            continue

        visited.add(state)
        nodes_expanded += 1
        max_depth = max(max_depth, len(path))
        
        if report_progress:
            report_progress(nodes_expanded, max_depth)

        if state == goal_state:
            end_time = time.time()
            result = {
                "algorithm": f"A* ({heuristic})",
                "path": path,
                "cost": len(path),
                "nodes_expanded": nodes_expanded,
                "search_depth": max_depth,
                "running_time": end_time - start_time
            }
            return result

        for neighbor, move in get_neighbors(state):
            if neighbor not in visited:
                new_cost = cost + 1
                new_f = new_cost + heuristic_distance(neighbor, heuristic)
                heapq.heappush(priority_queue, (new_f, new_cost, neighbor, path + [move]))

    return None



class PuzzleGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("8-Puzzle Solver")
        self.root.configure(bg=BACKGROUND_COLOR)
        self.root.resizable(False, False)
        
        # State and solution
        self.state = goal_state  # Initialize with the goal state
        self.solution = None
        self.solving = False
        self.animation_speed = 500  # ms between moves
        
        # Create custom fonts
        self.title_font = font.Font(family="Helvetica", size=16, weight="bold")
        self.button_font = font.Font(family="Helvetica", size=14)
        self.stats_font = font.Font(family="Helvetica", size=10)
        
        # Create main frames
        self.main_frame = tk.Frame(root, bg=BACKGROUND_COLOR)
        self.main_frame.pack(padx=20, pady=20)
        
        # Top row with title
        self.title_label = tk.Label(
            self.main_frame, 
            text="8-Puzzle Solver", 
            font=self.title_font, 
            bg=BACKGROUND_COLOR, 
            fg=TITLE_COLOR
        )
        self.title_label.grid(row=0, column=0, columnspan=2, pady=(0, 15))
        
        # Create left and right sections
        self.left_frame = tk.Frame(self.main_frame, bg=BACKGROUND_COLOR)
        self.left_frame.grid(row=1, column=0, padx=(0, 20))
        
        self.right_frame = tk.Frame(self.main_frame, bg=BACKGROUND_COLOR)
        self.right_frame.grid(row=1, column=1, sticky="n")
        
        # Create the puzzle grid
        self.create_grid()
        
        # Create custom state input
        self.create_state_input()
        
        # Create controls
        self.create_controls()
        
        # Create statistics display
        self.create_stats_display()
        
        # Update grid to show initial state
        self.update_grid()

    def create_state_input(self):
        state_frame = tk.LabelFrame(
            self.right_frame, 
            text="Custom State", 
            bg=BACKGROUND_COLOR, 
            fg=TEXT_COLOR,
            bd=2, 
            relief=tk.RIDGE
        )
        state_frame.pack(fill=tk.X, pady=10)
        
        # Instructions
        instruction_label = tk.Label(
            state_frame,
            text="Enter numbers 0-8 (0 is the empty space):",
            bg=BACKGROUND_COLOR,
            fg=TEXT_COLOR,
            anchor="w",
            justify=tk.LEFT,
            wraplength=250
        )
        instruction_label.pack(fill=tk.X, padx=10, pady=(5, 0))
        
        # Input frame for the 3x3 grid
        input_frame = tk.Frame(state_frame, bg=BACKGROUND_COLOR)
        input_frame.pack(padx=10, pady=5)
        
        # Create 3x3 grid of entry boxes
        self.state_entries = []
        for row in range(3):
            for col in range(3):
                entry = tk.Entry(
                    input_frame,
                    width=2,
                    font=self.button_font,
                    justify=tk.CENTER,
                    bd=1,
                    relief=tk.SOLID
                )
                entry.grid(row=row, column=col, padx=2, pady=2)
                self.state_entries.append(entry)
        
        # Buttons
        button_frame = tk.Frame(state_frame, bg=BACKGROUND_COLOR)
        button_frame.pack(fill=tk.X, padx=10, pady=(5, 10))
        
        self.apply_button = tk.Button(
            button_frame,
            text="Apply State",
            command=self.apply_custom_state,
            bg=ACCENT_COLOR,
            fg="white",
            activebackground=BUTTON_ACTIVE_COLOR,
            activeforeground="white",
            relief=tk.RAISED,
            bd=1,
            width=12
        )
        self.apply_button.pack(side=tk.LEFT, padx=(0, 5))
        
        self.clear_button = tk.Button(
            button_frame,
            text="Clear",
            command=self.clear_entries,
            bg=ACCENT_COLOR,
            fg="white",
            activebackground=BUTTON_ACTIVE_COLOR,
            activeforeground="white",
            relief=tk.RAISED,
            bd=1,
            width=12
        )
        self.clear_button.pack(side=tk.RIGHT)

    def apply_custom_state(self):
        if self.solving:
            return
            
        # Get values from entries
        values = []
        for entry in self.state_entries:
            val = entry.get().strip()
            if not val:
                messagebox.showerror("Error", "All cells must be filled")
                return
        
            try:
                num = int(val)
                if num < 0 or num > 8:
                    messagebox.showerror("Error", "Values must be between 0 and 8")
                    return
                values.append(str(num))
            except ValueError:
                messagebox.showerror("Error", "All values must be numbers")
                return
    
        # Check if all digits 0-8 are present exactly once
        if sorted(values) != ['0', '1', '2', '3', '4', '5', '6', '7', '8']:
            messagebox.showerror("Error", "Each digit from 0-8 must appear exactly once")
            return
    
        # Create new state string
        new_state = ''.join(values)
    
        
    
        # Apply the new state
        self.state = new_state
        self.update_grid()
        self.clear_stats()

    def clear_entries(self):
        for entry in self.state_entries:
            entry.delete(0, tk.END)

    def update_entries_from_current_state(self):
        # Update entry boxes to match current state
        for i, digit in enumerate(self.state):
            self.state_entries[i].delete(0, tk.END)
            self.state_entries[i].insert(0, digit)

    def create_grid(self):
        grid_frame = tk.Frame(self.left_frame, bg=BACKGROUND_COLOR, bd=2, relief=tk.RIDGE)
        grid_frame.pack(pady=10)
        
        self.tile_size = 80
        self.canvas = tk.Canvas(
            grid_frame, 
            width=self.tile_size * 3, 
            height=self.tile_size * 3,
            bg=BACKGROUND_COLOR,
            bd=0, 
            highlightthickness=0
        )
        self.canvas.pack()
        
        # Create tiles
        self.tiles = {}
        for i in range(9):
            row, col = divmod(i, 3)
            x, y = col * self.tile_size, row * self.tile_size
            
            # Create tile (rectangle and text)
            rect = self.canvas.create_rectangle(
                x, y, x + self.tile_size, y + self.tile_size,
                fill=BUTTON_COLOR if self.state[i] != '0' else EMPTY_COLOR,
                outline=BACKGROUND_COLOR,
                width=2
            )
            
            text = self.canvas.create_text(
                x + self.tile_size/2, y + self.tile_size/2,
                text=self.state[i] if self.state[i] != '0' else "",
                font=self.button_font
            )
            
            self.tiles[i] = (rect, text)
            
            # Add click event for manual moves
            self.canvas.tag_bind(rect, "<Button-1>", lambda e, idx=i: self.try_move_tile(idx))
            self.canvas.tag_bind(text, "<Button-1>", lambda e, idx=i: self.try_move_tile(idx))

    def try_move_tile(self, idx):
        if self.solving:
            return
            
        # Find zero position
        zero_pos = self.state.index('0')
        row, col = divmod(idx, 3)
        zero_row, zero_col = divmod(zero_pos, 3)
        
        # Check if this tile is adjacent to the empty space
        if ((abs(row - zero_row) == 1 and col == zero_col) or 
            (abs(col - zero_col) == 1 and row == zero_row)):
            # Swap tiles
            new_state = list(self.state)
            new_state[idx], new_state[zero_pos] = new_state[zero_pos], new_state[idx]
            self.state = ''.join(new_state)
            self.update_grid()
            self.update_entries_from_current_state()  # Update entry boxes to match new state
            
            # Check if solved
            if self.state == goal_state:
                messagebox.showinfo("Success", "Puzzle solved!")

    def update_grid(self):
        for i in range(9):
            rect, text = self.tiles[i]
            
            # Update color
            fill_color = BUTTON_COLOR if self.state[i] != '0' else EMPTY_COLOR
            self.canvas.itemconfig(rect, fill=fill_color)
            
            # Update text
            self.canvas.itemconfig(text, text=self.state[i] if self.state[i] != '0' else "")

    def create_controls(self):
        controls_frame = tk.LabelFrame(
            self.right_frame, 
            text="Controls", 
            bg=BACKGROUND_COLOR, 
            fg=TEXT_COLOR,
            bd=2, 
            relief=tk.RIDGE
        )
        controls_frame.pack(fill=tk.X, pady=10)
        
        # Algorithm selection
        tk.Label(
            controls_frame, 
            text="Algorithm:", 
            bg=BACKGROUND_COLOR, 
            fg=TEXT_COLOR
        ).grid(row=0, column=0, sticky="w", padx=10, pady=5)
        
        self.algo_var = tk.StringVar(value="BFS")
        algorithms = ["BFS", "DFS", "Iterative DFS", "A* Manhattan", "A* Euclidean"]
        algo_dropdown = ttk.Combobox(
            controls_frame, 
            textvariable=self.algo_var, 
            values=algorithms, 
            state="readonly",
            width=15
        )
        algo_dropdown.grid(row=0, column=1, sticky="we", padx=10, pady=5)
        algo_dropdown.bind("<<ComboboxSelected>>", self.toggle_max_depth)
        
        # Max depth for IDDFS
        self.max_depth_frame = tk.Frame(controls_frame, bg=BACKGROUND_COLOR)
        self.max_depth_frame.grid(row=1, column=0, columnspan=2, sticky="we", padx=10, pady=5)
        
        tk.Label(
            self.max_depth_frame, 
            text="Max Depth:", 
            bg=BACKGROUND_COLOR, 
            fg=TEXT_COLOR
        ).pack(side=tk.LEFT)
        
        self.max_depth_var = tk.StringVar(value="20")
        self.max_depth_slider = tk.Scale(
            self.max_depth_frame,
            from_=5,
            to=50,
            orient=tk.HORIZONTAL,
            variable=self.max_depth_var,
            bg=BACKGROUND_COLOR,
            highlightthickness=0,
            length=150
        )
        self.max_depth_slider.pack(side=tk.LEFT, padx=(5, 0), fill=tk.X, expand=True)
        
        # Initially hide max depth if not using IDDFS
        if self.algo_var.get() != "Iterative DFS":
            self.max_depth_frame.grid_remove()
        
        # Animation speed
        speed_frame = tk.Frame(controls_frame, bg=BACKGROUND_COLOR)
        speed_frame.grid(row=2, column=0, columnspan=2, sticky="we", padx=10, pady=5)
        
        tk.Label(
            speed_frame, 
            text="Animation Speed:", 
            bg=BACKGROUND_COLOR, 
            fg=TEXT_COLOR
        ).pack(side=tk.LEFT)
        
        self.speed_var = tk.IntVar(value=500)
        speed_slider = tk.Scale(
            speed_frame,
            from_=100,
            to=1000,
            orient=tk.HORIZONTAL,
            variable=self.speed_var,
            command=self.update_animation_speed,
            bg=BACKGROUND_COLOR,
            highlightthickness=0,
            length=150
        )
        speed_slider.pack(side=tk.LEFT, padx=(5, 0), fill=tk.X, expand=True)
        
        # Action buttons frame
        button_frame = tk.Frame(controls_frame, bg=BACKGROUND_COLOR)
        button_frame.grid(row=3, column=0, columnspan=2, sticky="we", padx=10, pady=10)
        
        # Shuffle button
        self.shuffle_button = tk.Button(
            button_frame,
            text="Shuffle",
            command=self.shuffle,
            bg=ACCENT_COLOR,
            fg="white",
            activebackground=BUTTON_ACTIVE_COLOR,
            activeforeground="white",
            relief=tk.RAISED,
            bd=1,
            width=10
        )
        self.shuffle_button.pack(side=tk.LEFT, padx=(0, 5))
        
        # Reset button
        self.reset_button = tk.Button(
            button_frame,
            text="Reset",
            command=self.reset_to_goal,
            bg=ACCENT_COLOR,
            fg="white",
            activebackground=BUTTON_ACTIVE_COLOR,
            activeforeground="white",
            relief=tk.RAISED,
            bd=1,
            width=10
        )
        self.reset_button.pack(side=tk.RIGHT)
        
        # Solve button
        self.solve_button = tk.Button(
            controls_frame,
            text="Solve Puzzle",
            command=self.solve,
            bg=ACCENT_COLOR,
            fg="white",
            activebackground=BUTTON_ACTIVE_COLOR,
            activeforeground="white",
            relief=tk.RAISED,
            bd=1,
            pady=5
        )
        self.solve_button.grid(row=4, column=0, columnspan=2, sticky="we", padx=10, pady=10)

    def create_stats_display(self):
        stats_frame = tk.LabelFrame(
            self.right_frame, 
            text="Statistics", 
            bg=BACKGROUND_COLOR, 
            fg=TEXT_COLOR,
            bd=2, 
            relief=tk.RIDGE
        )
        stats_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Progress indicators
        progress_frame = tk.Frame(stats_frame, bg=BACKGROUND_COLOR)
        progress_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Nodes expanded
        tk.Label(
            progress_frame, 
            text="Nodes expanded:", 
            bg=BACKGROUND_COLOR, 
            fg=TEXT_COLOR,
            anchor="w"
        ).grid(row=0, column=0, sticky="w")
        
        self.nodes_var = tk.StringVar(value="0")
        tk.Label(
            progress_frame, 
            textvariable=self.nodes_var, 
            bg=BACKGROUND_COLOR, 
            fg=TEXT_COLOR,
            anchor="e",
            width=10
        ).grid(row=0, column=1, sticky="e")
        
        # Current depth
        tk.Label(
            progress_frame, 
            text="Current depth:", 
            bg=BACKGROUND_COLOR, 
            fg=TEXT_COLOR,
            anchor="w"
        ).grid(row=1, column=0, sticky="w")
        
        self.depth_var = tk.StringVar(value="0")
        tk.Label(
            progress_frame, 
            textvariable=self.depth_var, 
            bg=BACKGROUND_COLOR, 
            fg=TEXT_COLOR,
            anchor="e",
            width=10
        ).grid(row=1, column=1, sticky="e")
        
        # Results
        self.result_text = tk.Text(
            stats_frame, 
            bg=BACKGROUND_COLOR, 
            fg=TEXT_COLOR,
            height=8, 
            width=30,
            relief=tk.FLAT,
            font=self.stats_font,
            state=tk.DISABLED
        )
        self.result_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

    def toggle_max_depth(self, *args):
        if self.algo_var.get() == "Iterative DFS":
            self.max_depth_frame.grid()
        else:
            self.max_depth_frame.grid_remove()

    def update_animation_speed(self, *args):
        self.animation_speed = self.speed_var.get()

    def generate_random_state(self):
            numbers = list("012345678")  
            random.shuffle(numbers)  
            return "".join(numbers)
    
    def shuffle(self):
        if self.solving:
            return
            
        self.state = self.generate_random_state()  
        self.update_grid()
        self.update_entries_from_current_state()  
        self.clear_stats()

    def reset_to_goal(self):
        if self.solving:
            return
            
        self.state = goal_state
        self.update_grid()
        self.update_entries_from_current_state()  # Update entry boxes to match new state
        self.clear_stats()

    def clear_stats(self):
        self.nodes_var.set("0")
        self.depth_var.set("0")
        self.result_text.config(state=tk.NORMAL)
        self.result_text.delete(1.0, tk.END)
        self.result_text.config(state=tk.DISABLED)

    def update_progress(self, nodes, depth):
        self.nodes_var.set(str(nodes))
        self.depth_var.set(str(depth))
        self.root.update_idletasks()

    def solve(self):
        if self.solving:
            return
            
        if self.state == goal_state:
            messagebox.showinfo("Already Solved", "The puzzle is already in the goal state!")
            return
            
       
            
        # Disable buttons during solving
        self.solve_button.config(state=tk.DISABLED)
        self.shuffle_button.config(state=tk.DISABLED)
        self.reset_button.config(state=tk.DISABLED)
        self.apply_button.config(state=tk.DISABLED)
        self.clear_button.config(state=tk.DISABLED)
        
        # Set solving flag
        self.solving = True
        
        # Clear previous results
        self.clear_stats()
        
        # Start solving in a separate thread
        algorithm = self.algo_var.get()
        t = threading.Thread(target=self.run_algorithm, args=(algorithm,))
        t.daemon = True
        t.start()

    def run_algorithm(self, algorithm):
        try:
            if algorithm == "BFS":
                result = bfs(self.state, self.update_progress)
            elif algorithm == "DFS":
                result = dfs(self.state, self.update_progress)
            elif algorithm == "Iterative DFS":
                try:
                    max_depth = int(self.max_depth_var.get())
                    if max_depth <= 0:
                        self.root.after(0, lambda: messagebox.showerror("Error", "Max depth must be a positive integer"))
                        self.finish_solving()
                        return
                    result = iterative_dfs(self.state, max_depth, self.update_progress)
                except ValueError:
                    self.root.after(0, lambda: messagebox.showerror("Error", "Max depth must be a valid integer"))
                    self.finish_solving()
                    return
            elif algorithm == "A* Manhattan":
                result = a_star(self.state, "manhattan", self.update_progress)
            elif algorithm == "A* Euclidean":
                result = a_star(self.state, "euclidean", self.update_progress)
            else:
                self.root.after(0, lambda: messagebox.showerror("Error", "Invalid Algorithm Selection"))
                self.finish_solving()
                return
                
            if result:
                self.solution = result
                self.root.after(0, self.display_results)
                self.root.after(0, self.animate_solution)
            else:
                self.root.after(0, lambda: messagebox.showerror("Error", "No solution found!"))
                self.finish_solving()
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Error", f"An error occurred: {str(e)}"))
            self.finish_solving()

    def display_results(self):
        if not self.solution:
            return
            
        # Update results in the text widget
        self.result_text.config(state=tk.NORMAL)
        self.result_text.delete(1.0, tk.END)
        
        self.result_text.insert(tk.END, f"Algorithm: {self.solution['algorithm']}\n")
        self.result_text.insert(tk.END, f"Path length: {self.solution['cost']}\n")
        self.result_text.insert(tk.END, f"Nodes expanded: {self.solution['nodes_expanded']}\n")
        self.result_text.insert(tk.END, f"Max search depth: {self.solution['search_depth']}\n")
        self.result_text.insert(tk.END, f"Running time: {self.solution['running_time']:.4f} sec\n")
                                                         
        self.result_text.config(state=tk.DISABLED)

    def animate_solution(self):
        if not self.solution or not self.solving:
            return
            
        self.animate_step(0)

    def animate_step(self, step_index):
        if not self.solving or step_index >= len(self.solution["path"]):
            self.finish_solving()
            return
            
        
            
        # Get the next move
        move = self.solution["path"][step_index]
        
        # Find the tile to move
        zero_index = find_zero_position(self.state)
        zero_row, zero_col = divmod(zero_index, 3)
        
        # Calculate the new position based on the move
        if move == "Up":
            new_row, new_col = zero_row - 1, zero_col
        elif move == "Down":
            new_row, new_col = zero_row + 1, zero_col
        elif move == "Left":
            new_row, new_col = zero_row, zero_col - 1
        elif move == "Right":
            new_row, new_col = zero_row, zero_col + 1
        
        new_index = new_row * 3 + new_col
        

        new_state = list(self.state)
        new_state[zero_index], new_state[new_index] = new_state[new_index], new_state[zero_index]
        self.state = ''.join(new_state)
        

        self.update_grid()
        

        self.root.after(self.animation_speed, lambda: self.animate_step(step_index + 1))

    def toggle_animation(self):
        if not self.solving:
            return
            
        

    def stop_animation(self):
        self.finish_solving()

    def finish_solving(self):
        self.solving = False
        
        
        # Re-enable buttons
        self.solve_button.config(state=tk.NORMAL)
        self.shuffle_button.config(state=tk.NORMAL)
        self.reset_button.config(state=tk.NORMAL)
        
       
        
        

if __name__ == "__main__":
    root = tk.Tk()
    app = PuzzleGUI(root)
    root.mainloop()