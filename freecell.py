import random
from queue import PriorityQueue
from copy import deepcopy

# カードのスートと数字を定義
SUITS = ['H', 'D', 'C', 'S']
RANKS = ['A', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K']

SAMPLE_TABLEAU = [
    [('8','H'),('5','D'),('9','D'),('5','C'),('4','C'),('2','H'),('9','C')],
    [('8','C'),('10','C'),('2','D'),('6','C'),('3','D'),('2','C'),('6','D')],
    [('Q','H'),('5','S'),('J','D'),('10','D'),('A','H'),('6','S'),('3','H')],
    [('J','S'),('5','H'),('8','D'),('10','S'),('3','S'),('7','D'),('7','S')],
    [('8','S'),('A','S'),('2','S'),('K','H'),('K','S'),('6','H')],
    [('A','D'),('7','C'),('K','C'),('Q','S'),('3','C'),('A','C')],
    [('4','H'),('Q','C'),('9','S'),('J','H'),('7','H'),('4','D')],
    [('J','C'),('K','D'),('9','H'),('4','S'),('Q','D'),('10','H')],
]

def is_valid_sequence(cards):
    """
    Check if the given list of cards form a valid descending sequence of alternating colors.
    """
    for i in range(1, len(cards)):
        current_card = cards[i - 1]
        next_card = cards[i]
        current_color = 'Red' if current_card[1] in ['H', 'D'] else 'Black'
        next_color = 'Red' if next_card[1] in ['H', 'D'] else 'Black'
        
        # Check for alternating colors and descending ranks
        if current_color == next_color or rank_value(current_card[0]) != rank_value(next_card[0]) + 1:
            return False
    return True

def move_tableau_to_tableau_final_v3(board, from_index, to_index, num_cards):
    """
    Move a card (or a sequence of cards) from one tableau to another, ensuring all rules are followed.
    This version ensures the moved sequence is a valid descending sequence of alternating colors.
    """
    # Create a deep copy of the board to avoid modifying the original
    board_copy = deepcopy(board)
    moving_cards = board_copy['tableau'][from_index][-num_cards:]

    # Ensure the cards being moved form a valid sequence
    if not is_valid_sequence(moving_cards):
        return board

    # If destination tableau is empty, we can move any cards
    if not board_copy['tableau'][to_index]:
        board_copy['tableau'][from_index] = board_copy['tableau'][from_index][:-num_cards]
        board_copy['tableau'][to_index] = moving_cards
        return board_copy

    # If destination tableau is not empty, ensure compatibility
    top_card_dest = board_copy['tableau'][to_index][-1]
    bottom_card_moving = moving_cards[0]
    
    # Check if the bottom card of the moving sequence is compatible with the top card of the destination tableau
    if (top_card_dest[1] in ['H', 'D'] and bottom_card_moving[1] in ['C', 'S'] or
            top_card_dest[1] in ['C', 'S'] and bottom_card_moving[1] in ['H', 'D']) and \
            rank_value(top_card_dest[0]) == rank_value(bottom_card_moving[0]) + 1:
        board_copy['tableau'][from_index] = board_copy['tableau'][from_index][:-num_cards]
        board_copy['tableau'][to_index].extend(moving_cards)
        return board_copy

    return board

def move_tableau_to_homecell_updated(board, tableau_index):
    """
    Move the top card from the specified tableau column to its corresponding homecell.
    Return a new board state if the move is valid, otherwise return the original board.
    """
    # Clone the board to avoid mutating the original state
    new_board = {
        'freecells': board['freecells'].copy(),
        'home_cells': [cell.copy() for cell in board['home_cells']],
        'tableau': [column.copy() for column in board['tableau']]
    }

    # Ensure the tableau index is valid
    if not (0 <= tableau_index < 8):
        return board

    # Get the moving card from the end of the specified tableau column
    moving_card = new_board['tableau'][tableau_index][-1] if new_board['tableau'][tableau_index] else None

    # If no card exists, return the original board
    if not moving_card:
        return board

    # Determine the expected rank for the moving card based on the current homecell
    suit_index = SUITS.index(moving_card[1])
    expected_rank_index = len(new_board['home_cells'][suit_index])

    # If the moving card's rank matches the expected rank, move the card
    if RANKS.index(moving_card[0]) == expected_rank_index:
        new_board['home_cells'][suit_index].append(moving_card)
        new_board['tableau'][tableau_index].pop()

    return new_board

def move_tableau_to_freecell(board, tableau_index):
    """
    Move the top card from the specified tableau column to an available freecell.
    Return a new board state if the move is valid, otherwise return the original board.
    """
    # Clone the board to avoid mutating the original state
    new_board = {
        'freecells': board['freecells'].copy(),
        'home_cells': [cell.copy() for cell in board['home_cells']],
        'tableau': [column.copy() for column in board['tableau']]
    }

    # Ensure the tableau index is valid
    if not (0 <= tableau_index < 8):
        return board

    # Get the moving card from the end of the specified tableau column
    moving_card = new_board['tableau'][tableau_index][-1] if new_board['tableau'][tableau_index] else None

    # If no card exists, return the original board
    if not moving_card:
        return board

    # Check if there's an available freecell
    if len(new_board['freecells']) < 4:
        new_board['freecells'].append(moving_card)
        new_board['tableau'][tableau_index].pop()
    else:
        # No available freecell, return the original board
        return board

    return new_board

def move_freecell_to_tableau(board, freecell_index, tableau_index):
    """
    Move the specified card from the freecells to the end of the specified tableau column.
    Return a new board state if the move is valid, otherwise return the original board.
    """
    # Clone the board to avoid mutating the original state
    new_board = {
        'freecells': board['freecells'].copy(),
        'home_cells': [cell.copy() for cell in board['home_cells']],
        'tableau': [column.copy() for column in board['tableau']]
    }

    # Ensure the freecell and tableau indices are valid
    if not (0 <= freecell_index < len(new_board['freecells']) and 0 <= tableau_index < 8):
        return board

    # Get the moving card from the specified freecell
    moving_card = new_board['freecells'][freecell_index]

    # If the destination tableau is empty, move the card. Otherwise, check for validity.
    if not new_board['tableau'][tableau_index]:
        new_board['tableau'][tableau_index].append(moving_card)
        new_board['freecells'].remove(moving_card)
    else:
        top_card_dest = new_board['tableau'][tableau_index][-1]

        # Check if the cards are of opposite colors and in descending order
        if (top_card_dest[1] in ['H', 'D'] and moving_card[1] in ['C', 'S'] or
            top_card_dest[1] in ['C', 'S'] and moving_card[1] in ['H', 'D']):
            rank_index_dest = RANKS.index(top_card_dest[0])
            rank_index_moving = RANKS.index(moving_card[0])

            if rank_index_dest == rank_index_moving + 1:
                new_board['tableau'][tableau_index].append(moving_card)
                new_board['freecells'].remove(moving_card)
            else:
                return board
        else:
            return board

    return new_board

def move_freecell_to_homecell(board, freecell_index):
    """
    Move the specified card from the freecells to its corresponding homecell.
    Return a new board state if the move is valid, otherwise return the original board.
    """
    # Clone the board to avoid mutating the original state
    new_board = {
        'freecells': board['freecells'].copy(),
        'home_cells': [cell.copy() for cell in board['home_cells']],
        'tableau': [column.copy() for column in board['tableau']]
    }

    # Ensure the freecell index is valid
    if not (0 <= freecell_index < len(new_board['freecells'])):
        return board

    # Get the moving card from the specified freecell
    moving_card = new_board['freecells'][freecell_index]

    # Determine the expected rank for the moving card based on the current homecell
    suit_index = SUITS.index(moving_card[1])
    expected_rank_index = len(new_board['home_cells'][suit_index])

    # If the moving card's rank matches the expected rank, move the card
    if RANKS.index(moving_card[0]) == expected_rank_index:
        new_board['home_cells'][suit_index].append(moving_card)
        new_board['freecells'].remove(moving_card)
    else:
        return board

    return new_board




def print_board(board):
    """
    Print the board state with a space between cards.
    """
    # Print freecells
    freecells_display = ", ".join([f"{card[0]:<2}{card[1]}" for card in board['freecells']])
    print(f"Freecells: [{freecells_display}]")

    # Print homecells
    home_cells_display = ", ".join([f"{card[-1][0]:<2}{suit}" if card else "__" for suit, card in zip(SUITS, board['home_cells'])])
    print(f"Homecells: [{home_cells_display}]")

    # Print tableau
    print("\nTableau:")
    max_tableau_length = max(len(col) for col in board['tableau'])
    for i in range(max_tableau_length):
        row_display = []
        for col in board['tableau']:
            row_display.append(f"{col[i][0]:<2}{col[i][1]} " if i < len(col) else "    ")
        print(" ".join(row_display))

def heuristic(board):
    """
    Compute a heuristic value for the given board state.
    Lower values indicate states closer to the goal.
    """
    # Number of cards in home_cells (the more the better)
    home_cells_count = sum(len(cell) for cell in board['home_cells'])

    # Number of free spaces in freecells (the more the better)
    free_spaces = 4 - len(board['freecells'])

    # Number of empty tableau columns (the more the better)
    empty_tableau_cols = sum(1 for col in board['tableau'] if not col)

    # Our heuristic is a combination of the above factors
    # We give more weight to home_cells_count since it directly contributes to the goal
    return -home_cells_count - 0.5 * free_spaces - 0.5 * empty_tableau_cols

def is_valid_sequence_corrected(cards):
    """
    Check if the given list of cards form a valid descending sequence of alternating colors.
    This corrected version explicitly checks the card sequence and colors.
    """
    for i in range(1, len(cards)):
        current_card = cards[i - 1]
        next_card = cards[i]
        current_rank = rank_value(current_card[0])
        next_rank = rank_value(next_card[0])
        
        current_color = 'Red' if current_card[1] in ['H', 'D'] else 'Black'
        next_color = 'Red' if next_card[1] in ['H', 'D'] else 'Black'
        
        # Check for descending sequence and alternating colors
        if not (current_rank - 1 == next_rank and current_color != next_color):
            return False
    return True

# Test the corrected is_valid_sequence function with the provided sequence



# Adjusting the get_possible_moves_corrected_v6 function to use the corrected is_valid_sequence function
def get_possible_moves_corrected_v7(board):
    """
    Generate all possible next board states from the current board state.
    This version uses the corrected is_valid_sequence function.
    """
    moves = []

    # Try moving cards between tableau columns
    for i in range(8):
        for j in range(8):
            if i != j:
                for num_cards in range(1, len(board['tableau'][i]) + 1):
                    moving_cards = board['tableau'][i][-num_cards:]
                    
                    # Ensure that the cards being moved are in a valid sequence
                    if not is_valid_sequence_corrected(moving_cards):
                        continue
                    
                    new_board = move_tableau_to_tableau_final_v3(board, i, j, num_cards)
                    if new_board != board:
                        moves.append((new_board, f"Move {num_cards} cards from tableau {i} to tableau {j}"))

    # Other moves remain the same as before
    for i in range(8):
        new_board = move_tableau_to_homecell_updated(board, i)
        if new_board != board:
            card = board['tableau'][i][-1]
            moves.append((new_board, f"Move {card[0]}{card[1]} from tableau {i} to home cell"))

    for i in range(8):
        new_board = move_tableau_to_freecell(board, i)
        if new_board != board:
            card = board['tableau'][i][-1]
            moves.append((new_board, f"Move {card[0]}{card[1]} from tableau {i} to freecell"))

    for i, card in enumerate(board['freecells']):
        for j in range(8):
            new_board = move_freecell_to_tableau(board, i, j)
            if new_board != board:
                moves.append((new_board, f"Move {card[0]}{card[1]} from freecell to tableau {j}"))

    for i, card in enumerate(board['freecells']):
        new_board = move_freecell_to_homecell(board, i)
        if new_board != board:
            moves.append((new_board, f"Move {card[0]}{card[1]} from freecell to home cell"))

    return moves



def a_star_search_final(initial_board):
    """
    Find a solution for the given FreeCell board using A* search.
    """
    # Each state in the priority queue is represented as (priority, board_str, board, moves)
    # Where 'moves' is the sequence of moves to reach the current board state
    start_state = (heuristic(initial_board), str(initial_board), initial_board, [])
    frontier = PriorityQueue()
    frontier.put(start_state)

    explored = set()

    while not frontier.empty():
        _, _, current_board, moves = frontier.get()

        # Check if we have already explored this board state
        board_str = str(current_board)
        if board_str in explored:
            continue
        explored.add(board_str)

        # Check if the current board state is a goal state (all cards are in the home cells)
        if all(len(cell) == 13 for cell in current_board['home_cells']):
            return moves  # We found a solution

        # Expand the current board state by trying all possible moves
        for move in get_possible_moves_updated(current_board):
            new_board, move_description = move
            new_moves = moves + [move_description]
            priority = heuristic(new_board) + len(new_moves)  # f = g + h
            frontier.put((priority, str(new_board), new_board, new_moves))

    return None  # No solution found

def rank_value(rank):
    """Convert card rank to its numeric value for heuristic evaluation."""
    if rank == 'A':
        return 1
    elif rank == 'J':
        return 11
    elif rank == 'Q':
        return 12
    elif rank == 'K':
        return 13
    else:
        return int(rank)

def heuristic_updated(board):
    """
    Compute an improved heuristic value for the given board state.
    """
    # Number of cards in home_cells (the more the better)
    home_cells_count = sum(len(cell) for cell in board['home_cells'])

    # Top card rank in home_cells (higher rank is better)
    home_cells_rank = sum(rank_value(cell[-1][0]) if cell else 0 for cell in board['home_cells'])

    # Number of free spaces in freecells (the more the better)
    free_spaces = 4 - len(board['freecells'])

    # Number of empty tableau columns (the more the better)
    empty_tableau_cols = sum(1 for col in board['tableau'] if not col)

    # Count of continuous sequences in tableau
    continuous_sequences = 0
    for col in board['tableau']:
        if len(col) > 1:
            for i in range(len(col) - 1):
                if rank_value(col[i][0]) - 1 == rank_value(col[i + 1][0]):
                    continuous_sequences += 1

    # Our updated heuristic is a combination of the above factors
    return (-home_cells_count * 10
            - home_cells_rank * 5
            - continuous_sequences * 3
            - free_spaces * 2
            - empty_tableau_cols)



def a_star_search_improved(initial_board):
    """
    Find a solution for the given FreeCell board using A* search with improved heuristic.
    """
    # Each state in the priority queue is represented as (priority, board_str, board, moves)
    # Where 'moves' is the sequence of moves to reach the current board state
    start_state = (heuristic_updated(initial_board), str(initial_board), initial_board, [])
    frontier = PriorityQueue()
    frontier.put(start_state)

    explored = set()

    while not frontier.empty():
        _, _, current_board, moves = frontier.get()

        # Check if we have already explored this board state
        board_str = str(current_board)
        if board_str in explored:
            continue
        explored.add(board_str)

        # Check if the current board state is a goal state (all cards are in the home cells)
        if all(len(cell) == 13 for cell in current_board['home_cells']):
            return moves  # We found a solution

        # Expand the current board state by trying all possible moves
        for move in get_possible_moves_corrected_v7(current_board):
            new_board, move_description = move
            new_moves = moves + [move_description]
            priority = heuristic_updated(new_board) + len(new_moves)  # f = g + h
            frontier.put((priority, str(new_board), new_board, new_moves))

    return None  # No solution found



if __name__ == "__main__":
    # 初期状態のフリーセルを設定
    # 例: {'freecells': [], 'home_cells': [[], [], [], []], 'tableau': [[('10', 'H'), ('9', 'D'), ...], [...], [...], [...]]}
    initial_board = {
        'freecells': [],
        'home_cells': [[], [], [], []],
        'tableau': SAMPLE_TABLEAU,
    }
    solution_moves_improved = a_star_search_improved(initial_board)
    print(solution_moves_improved)
    # print_board(initial_board)  # 初期状態をログとして出力
    # a_star_search_final(initial_board)
    # solve_freecell(initial_board)
