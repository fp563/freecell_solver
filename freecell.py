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

INITIAL_BOARL_SAMPLE = {
        'freecells': [],
        'home_cells': [[], [], [], []],
        'tableau': SAMPLE_TABLEAU,
    }

TEST_BOARD = {
        'freecells': [('6','D'),('2','D'),('3','D'),('4','D')],
        'home_cells': [[('1','H'),('2','H'),('3','H')],[],[('1','C'),('2','C'),('3','C'),('4','C'),('5','C'),('6','C')], []],
        'tableau': [
            [],
            [('8','C'),('10','C'),('9','D')],
            [('Q','H'),('5','S'),('J','D'),('10','D')],
            [('J','S'),('5','H'),('8','D'),('10','S'),('3','S'),('7','D')],
            [('8','S'),('A','S'),('2','S'),('K','H'),('K','S')],
            [('A','D'),('7','C'),('K','C'),('Q','S')],
            [('4','H'),('Q','C'),('9','S'),('J','H'),('7','H'),('6','S'),('5','D')],
            [('J','C'),('K','D'),('9','H'),('4','S'),('Q','D'),('10','H'),('9','C'),('8','H'),('7','S'),('6','H')],
        ],
    }


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

def move_tableau_to_tableau(board, from_index, to_index, num_cards):
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

def move_tableau_to_homecell(board, tableau_index):
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


def is_valid_sequence(cards):
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
def get_possible_moves(board):
    """
    Generate all possible next board states from the current board state.
    This version uses the corrected is_valid_sequence function.
    """
    moves = []

    # Calculate the maximum number of cards that can be moved in a sequence
    # based on the number of empty freecells and tableau columns.
    num_empty_freecells = 4 - len(board['freecells'])
    num_empty_tableau_cols = sum(1 for col in board['tableau'] if not col)

    
    # For normal moves between non-empty tableau columns
    max_movable_cards = (num_empty_freecells + 1) * (2 ** num_empty_tableau_cols)
    # For moving to an empty tableau column
    max_movable_to_empty_col = (num_empty_freecells + 1) * (2 ** (num_empty_tableau_cols-1))

    # Try moving cards between tableau columns
    for i in range(8):
        for j in range(8):
            if i != j:
                # If the destination tableau column is empty
                if not board['tableau'][j]:
                    limit = min(len(board['tableau'][i]) + 1, max_movable_to_empty_col + 1)
                else:
                    limit = min(len(board['tableau'][i]) + 1, max_movable_cards + 1)

                for num_cards in range(1, limit):
                    moving_cards = board['tableau'][i][-num_cards:]
                    
                    # Ensure that the cards being moved are in a valid sequence
                    if not is_valid_sequence(moving_cards):
                        continue
                    
                    new_board = move_tableau_to_tableau(board, i, j, num_cards)
                    if new_board != board:
                        moves.append((new_board, f"Move {num_cards} cards from tableau {i} to tableau {j}"))

    # Other moves remain the same as before
    for i in range(8):
        new_board = move_tableau_to_homecell(board, i)
        if new_board != board:
            card = board['tableau'][i][-1]
            moves.append((new_board, f"Move {card[0]}{card[1]} from tableau {i} to home cell"))

    for i in range(8):
        new_board = move_tableau_to_freecell(board, i)
        if new_board != board:
            card = board['tableau'][i][-1]
            moves.append((new_board, f"Move {card[0]}{card[1]} from tableau {i} to free cell"))

    for i, card in enumerate(board['freecells']):
        for j in range(8):
            new_board = move_freecell_to_tableau(board, i, j)
            if new_board != board:
                moves.append((new_board, f"Move {card[0]}{card[1]} from free cell to tableau {j}"))

    for i, card in enumerate(board['freecells']):
        new_board = move_freecell_to_homecell(board, i)
        if new_board != board:
            moves.append((new_board, f"Move {card[0]}{card[1]} from free cell to home cell"))

    return moves

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

def count_cards_above_next_home_cards(board):
    total_count = 0
    
    for suit_index, home_cell in enumerate(board['home_cells']):
        target_card = None
        if home_cell:
            target_rank_index = RANKS.index(home_cell[-1][0]) + 1
            if target_rank_index < 13:
                target_card = (RANKS[target_rank_index], SUITS[suit_index])
        else:
            # If the homecell for the suit is empty, we look for 'A' of that suit
            target_card = ('A', SUITS[suit_index])
        
        if target_card:
            for col in board['tableau']:
                if target_card in col:
                    total_count += len(col) - col.index(target_card) - 1
                    break
                    
    return total_count

def minimum_max_rank_in_homecells(board):
    """
    Returns the smallest of the largest ranks among the 4 suits in the home cells.
    If any of the home cells is empty, returns 0.
    """
    max_ranks = []
    
    for home_cell in board['home_cells']:
        if not home_cell:  # If the home cell is empty
            return 0
        max_rank = rank_value(home_cell[-1][0])
        max_ranks.append(max_rank)
    
    return min(max_ranks)

def heuristic(board):
    """
    Compute an improved heuristic value for the given board state.
    """
    # Number of cards in home_cells (the more the better)
    home_cells_count = sum(len(cell) for cell in board['home_cells'])

    home_cells_suits = sum(1 if cell else 0 for cell in board['home_cells'])

    # Top card rank in home_cells (higher rank is better)
    home_cells_rank = sum(rank_value(cell[-1][0]) if cell else 0 for cell in board['home_cells'])

    # Number of free spaces in freecells (the more the better)
    free_spaces = 4 - len(board['freecells'])

    # Number of empty tableau columns (the more the better)
    empty_tableau_cols = sum(1 for col in board['tableau'] if not col)

    king_starting_cols = sum(1 for col in board['tableau'] if col and col[0][0] == 'K')

    penalty_gap_with_next_home = count_cards_above_next_home_cards(board)

    min_homecell_card_rank = minimum_max_rank_in_homecells(board)


    # Count of continuous sequences in tableau
    continuous_sequences_length = 0
    for col in board['tableau']:
        for length in range(len(col), 1, -1):
            if is_valid_sequence(col[-length:]):
                continuous_sequences_length += length
                break

    # Our updated heuristic is a combination of the above factors
    return (- home_cells_count * 70
            - home_cells_rank * 2
            - home_cells_suits * 100
            - min_homecell_card_rank * 100
            - continuous_sequences_length * 1
            - free_spaces * 5
            - empty_tableau_cols * 10
            - king_starting_cols * 20
            + penalty_gap_with_next_home * 5
            )

def string_to_card(str):
    if len(str)==3:
        return (str[0:2],str[2])
    else:
        return (str[0],str[1])


def print_board_state_after_each_move(initial_board, solution_moves):
    """
    Prints the board state after applying each move in the solution_moves list.
    """
    current_board = deepcopy(initial_board)

    print("Initial board state:")
    print({
        'freecells': current_board['freecells'],
        'home_cells': current_board['home_cells'],
        'tableau': current_board['tableau']
    })
    print("\n")

    for move in solution_moves:
        # Split the move description to extract move details
        move_parts = move.split()
        move_type_from = move_parts[-5]
        move_type_to = move_parts[-2]

        if move_type_from == "tableau":
            if move_type_to == "tableau":
                from_index = int(move_parts[-4])
                to_index = int(move_parts[-1])
                num_cards = int(move_parts[1])
                current_board = move_tableau_to_tableau(current_board, from_index, to_index, num_cards)
            elif move_type_to == "free":
                index = int(move_parts[-4])
                current_board = move_tableau_to_freecell(current_board, index)
            elif move_type_to == "home":
                index = int(move_parts[-4])
                current_board = move_tableau_to_homecell(current_board, index)
        elif move_type_from == "free":
            freecell_card = string_to_card(move_parts[1])
            freecell_index = current_board['freecells'].index(freecell_card)
            if move_type_to == "tableau":
                tableau_index = int(move_parts[-1])

                current_board = move_freecell_to_tableau(current_board, freecell_index, tableau_index)
            elif move_type_to == "home":
                current_board = move_freecell_to_homecell(current_board, freecell_index)

        print(f"After move: {move}")
        print({
            'freecells': current_board['freecells'],
            'home_cells': current_board['home_cells'],
            'tableau': current_board['tableau']
        })
        print("\n")

def a_star_search_improved(initial_board, max_iterations=1000):
    """
    Find a solution for the given FreeCell board using A* search with improved heuristic.
    """
    # Each state in the priority queue is represented as (priority, board_str, board, moves)
    # Where 'moves' is the sequence of moves to reach the current board state
    start_state = (heuristic(initial_board), str(initial_board), initial_board, [])
    frontier = PriorityQueue()
    frontier.put(start_state)

    explored = set()

    iterations = 0

    while not frontier.empty():
        _, _, current_board, moves = frontier.get()

        # Check if we have already explored this board state
        board_str = str(current_board)
        if board_str in explored:
            continue
        explored.add(board_str)

        # Check if the current board state is a goal state (all cards are in the home cells)
        if all(len(cell) == 13 for cell in current_board['home_cells']):
            print_board_state_after_each_move(initial_board, moves)
            return moves  # We found a solution

        # Expand the current board state by trying all possible moves
        for move in get_possible_moves(current_board):
            new_board, move_description = move
            new_moves = moves + [move_description]
            priority = heuristic(new_board) + len(new_moves)  # f = g + h
            frontier.put((priority, str(new_board), new_board, new_moves))

        iterations+=1
        if max_iterations and iterations >= max_iterations:
            return None

    return None  # No solution found



if __name__ == "__main__":
    # 初期状態のフリーセルを設定
    # 例: {'freecells': [], 'home_cells': [[], [], [], []], 'tableau': [[('10', 'H'), ('9', 'D'), ...], [...], [...], [...]]}
    initial_board = TEST_BOARD
    solution_moves_improved = a_star_search_improved(initial_board)
    print(solution_moves_improved)
