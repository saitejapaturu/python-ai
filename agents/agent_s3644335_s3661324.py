from connectfour.agents.computer_player import RandomAgent
import random
import math

# TeamName      :     skynet
# Team Members  :     Sri Sai Teja Paturu(s3644335), Janssen Wong(s3661324)


class StudentAgent(RandomAgent):
    def __init__(self, name):
        super().__init__(name)
        self.MaxDepth = 4

    def get_move(self, board):
        """
        Args:
            board: An instance of `Board` that is the current state of tshe board.

        Returns:
            A tuple of two integers, (row, col)
        """

        valid_moves = board.valid_moves()
        vals = []
        moves = []

        for move in valid_moves:
            next_state = board.next_state(self.id, move[1])
            moves.append(move)
            vals.append(self.dfMiniMax(next_state, 1, -math.inf, math.inf))

        bestMove = moves[vals.index(max(vals))]
        return bestMove

    def dfMiniMax(self, board, depth, a, b):
        # Goal return column with maximized scores of all possible next states

        # Setting Player and opponent
        # if player is 1, then enemy = (1%2) + 1 = 1+1 = 2
        # if player is 2, then enemy = (2%2) + 1 = 0+1 = 1
        player = self.id
        enemy = (player % 2) + 1

        # Check winning state
        winner = board.winner()
        if winner == player:
            return 99999
        elif winner == enemy:
            return -99999

        if depth == self.MaxDepth:
            return self.evaluateBoardState(board)

        valid_moves = board.valid_moves()
        vals = []
        moves = []

        for move in valid_moves:
            if depth % 2 == 1:
                next_state = board.next_state(self.id % 2 + 1, move[1])
            else:
                next_state = board.next_state(self.id, move[1])

            moves.append(move)
            vals.append(self.dfMiniMax(next_state, depth + 1, a, b))

            # alpha beta pruning
            if depth % 2 == 1:
                b = min(min(vals), b)
                if a >= b:
                    break
            else:
                a = max(a, max(vals))
                if a >= b:
                    break

        if depth % 2 == 1:
            # If the vals is empty, this happens when, the moves left on board < max depth
            # If so, returns a random value
            if not vals:
                return 1
            bestVal = min(vals)
        else:
            # If the vals is empty, this happens when, the moves left on board < max depth
            # If so, returns a random value
            if not vals:
                return 1
            bestVal = max(vals)

        return bestVal

    def evaluateBoardState(self, board):
        """
        Your evaluation function should look at the current state and return a score for it. 
        As an example, the random agent provided works as follows:
            If the opponent has won this game, return -1.
            If we have won the game, return 1.
            If neither of the players has won, return a random number.
        """

        """
        These are the variables and functions for board objects which may be helpful when creating your Agent.
        Look into board.py for more information/descriptions of each, or to look for any other definitions which may help you.

        Board Variables:
            board.width                 -->     always gives 7, even if one column is filled
            board.height                -->     doesn't work
            board.last_move             -->     last move placed, not useful as takes the move taken for evaluation.
            board.num_to_connect        -->     Always gives 4, cause connect 4
            board.winning_zones         -->     Not useful for assignment
            board.score_array           -->     Not useful for assignment
            board.current_player_score  -->     Always give [0, 0] unless updated

        Board Functions:
            board.get_cell_value(row, col)    -->   tell's weather the cell is occupied by enemy, us or none.
            board.try_move(col)               -->   Tells how many cells can be kept in a column -1.
            board.valid_move(row, col)        -->   tells if the move is valid.
            board.valid_moves()               -->   <generator object Board.valid_moves at 0x10c849d00>
            board.terminal(self)              -->   ?
            board.legal_moves()               -->   gives a list in which columns we can place at.
            board.next_state(turn)            -->   ? <connectfour.board.Board object at 0x10cba40b8>
            board.winner()                    -->   if a player won, returns the player id
        """
        score = 0
        num_to_connect = board.num_to_connect

        player = self.id
        enemy = (player % 2) + 1

        player_max_chains_formed = self.max_chains_formed(board, player)
        enemy_max_chains_formed = self.max_chains_formed(board, enemy)

        player_longest_chain = 0
        enemy_longest_chain = 0

        for num in range(num_to_connect):
            if player_max_chains_formed[num] > 0:
                player_longest_chain = (num+1)
            if enemy_max_chains_formed[num] > 0:
                enemy_longest_chain = (num+1)

        column = board.last_move[1]

        if player_longest_chain == 4:
            score = 99999
        elif enemy_longest_chain == 4:
            score = -99999
        else:
            for num in range(2, num_to_connect):
                score += player_max_chains_formed[num-1] * num * 100
                score -= enemy_max_chains_formed[num-1] * num * 10
            score += (self.column_priority(column)*10)

        if not self.guarantee_priority(board, player):
            return -1

        if self.guarantee_priority(board, enemy):
            score -= 10000

        if self.prevent_vertical_upscale(board):
            score = -99990

        return score

    def max_chains_formed(self, board, player):
        max_height = board.height - 1

        valid_moves = board.valid_moves()
        col_range = []

        for move in valid_moves:
            if move[0] < max_height:
                max_height = move[0]
            if move[0] != 5:
                col_range.append(move[1])

        # if the columns are totally full and the others are empty.
        if not col_range:
            col_range = list(range(0, board.width))

        row_range = list(range(max_height, board.height))
        max_height += 1

        horizontal_chains = self.horizontal_chains(board, row_range, col_range, player)
        vertical_chains = self.vertical_chains(board, row_range, col_range, player)
        diagonal_chains = self.diagonal_chains(board, row_range, col_range, player)

        max_1_chains = max(horizontal_chains[0], vertical_chains[0], diagonal_chains[0])
        max_2_chains = max(horizontal_chains[1], vertical_chains[1], diagonal_chains[1])
        max_3_chains = max(horizontal_chains[2], vertical_chains[2], diagonal_chains[2])
        max_4_chains = max(horizontal_chains[3], vertical_chains[3], diagonal_chains[3])

        max_chains_formed = [max_1_chains, max_2_chains, max_3_chains, max_4_chains]

        return max_chains_formed


    # Counts the number of 1-chain, 2-chain, 3-chain, 4-chain of the player in the board
    # returns chains[number of 1-chain, number of 2-chain, number of 3-chain, number of 4-chain]
    def horizontal_chains(self, board, row_range, col_range, player):
        chains = [0, 0, 0, 0]

        for row in row_range:
            chain = 0

            for col in col_range:
                if board.get_cell_value(row, col) == player:
                    if (col > 0) and (board.get_cell_value(row, (col-1)) == player):
                        chain += 1
                    else:
                        chain = 1
                else:
                    if chain != 0:
                        chains[chain-1] += 1
                        chain = 0

        return chains

    def vertical_chains(self, board, row_range, col_range, player):
        chains = [0, 0, 0, 0]

        for col in col_range:
            chain = 0

            for row in row_range:
                if board.get_cell_value(row, col) == player:
                    if (row > min(row_range)) and (board.get_cell_value((row-1), col) == player):
                        chain += 1

                    else:
                        chain = 1
                else:
                    if chain != 0:
                        chains[chain-1] += 1
                        chain = 0

        return chains

    def diagonal_chains(self, board, row_range, col_range, player):
        rl_chains = [0, 0, 0, 0]
        lr_chains = [0, 0, 0, 0]

        min_height = min(row_range)

        min_width = min(col_range)
        max_width = max(col_range)

        # Checks Left to Right
        for col in col_range:

            chain = 0
            for row in row_range:
                increment = (row - min_height)
                column = col + increment

                if column <= max_width:
                    if board.get_cell_value(row, column) == player:
                        if ((row > min_height) and (column > min_width) and
                                (board.get_cell_value((row-1), (column-1)) == player)):
                            chain += 1

                        else:
                            chain = 1
                    else:
                        if chain != 0:
                            lr_chains[chain - 1] += 1
                            chain = 0

        # Checks Right to Left
        for col in col_range:
            chain = 0
            for row in row_range:
                decrement = (row - min_height)
                column = col - decrement

                if column >= min_width:
                    if board.get_cell_value(row, column) == player:
                        if ((row > min_height) and (column < max_width) and
                                (board.get_cell_value((row-1), (column+1)) == player)):
                            chain += 1

                        else:
                            chain = 1
                    else:
                        if chain != 0:
                            rl_chains[chain - 1] += 1
                            chain = 0

        chains = [max(lr_chains[0], rl_chains[0]),
                  max(lr_chains[1], rl_chains[1]),
                  max(lr_chains[2], rl_chains[2]),
                  max(lr_chains[3], rl_chains[3]),]

        return chains

    def column_priority(self, col):
        score = 0
        if col == 3:
            score += 4
        elif col == 2 or col == 4:
            score += 3
        elif col == 1 or col == 5:
            score += 2

        if col >= 3:
            score += 1

        return score

    # Gets the current board state, player, max_chain length of the player
    # Checks all the direction, to make sure the move made will allow to connect 4.
    # Returns false, if the current move cannot lead to connect 4 chain.
    def guarantee_priority(self, board, player):

        move_guarantees_connect_4 = False

        last_move = board.last_move
        last_move_row = last_move[0]
        last_move_column = last_move[1]

        num_to_connect = board.num_to_connect

        # Check Vertical

        # Checking cells from  bottom of last move placed of the same column
        # and creating a chain of cells of same player until the placed cell
        chain = 0
        for row in range(board.height-1, last_move_row):
            if board.get_cell_value(row, last_move_column) == player:
                chain += 1
            else:
                chain = 0

        # since the last move is 1 step closer to connect 4
        chain += 1

        # Checks if there is enough height to make a connect 4 possible.
        if last_move_row >= (num_to_connect - chain):
            move_guarantees_connect_4 = True

        # Checks  Horizontal

        # Checking cells from  left of the board to last move placed
        # and creating a chain of cells of same player until the placed cell
        lr_chain = 0
        # Valid moves is used to check the depth of the columns
        # This allows to prevent the cases, when a cell is place on the top row
        # and the other cells at that level are empty
        valid_moves = board.valid_moves()
        for move in valid_moves:
            if move[1] < last_move_column:
                if move[0] == last_move_row-1 or board.get_cell_value(last_move_row, move[1]) == player:
                    lr_chain += 1
                else:
                    lr_chain = 0
            else:
                break


        # Checking cells from last move placed to the right of the board
        # Until the the top right cells diagonally are either empty or of the same player
        # spaces_used_to_win is built, then it breaks
        rl_chain = 0
        for move in valid_moves:
            if move[1] >= last_move_column:
                if move[0] == last_move_row-1 or board.get_cell_value(last_move_row ,move[1]) == player:
                    rl_chain +=1
                else:
                    break
            else:
                break

        # since the last move is 1 step closer to connect 4
        spaces_used_to_win = rl_chain + lr_chain + 1

        if spaces_used_to_win >= num_to_connect:
            move_guarantees_connect_4 = True

        # Check Diagonal Left to Right
        chain = 0

        # Checking cells from bottom left to top right with respect to last move
        # and creating a chain of cells of same player until the placed cell
        for row in range(board.height-1, last_move_row):
            step = last_move_row - row
            if board.get_cell_value(row, (last_move_column-step)) == player or 0:
                chain += 1
            else:
                chain = 0

        # Checks cells the top right direction diagonally with respect to last move
        # Until the the top right cells diagonally are either empty or of the same player
        # spaces_used_to_win is built, then it breaks
        spaces_used_to_win = 0
        for row in range(last_move_row+1, -1):
            step = last_move_row - row
            if board.get_cell_value(row, (last_move_column + step)) == player or 0:
                chain += 1
            else:
                break

        if (chain + spaces_used_to_win) >= num_to_connect:
            move_guarantees_connect_4 = True

        # Check Diagonal Right to Left
        chain = 0

        # Checking cells from bottom right to top left with respect to last move
        # and creating a chain of cells of same player until the placed cell
        for row in range(board.height - 1, last_move_row):
            step = last_move_row - row
            if board.get_cell_value(row, (last_move_column + step)) == player or 0:
                chain += 1
            else:
                chain = 0

        # Checks cells the top left direction diagonally with respect to last move
        # Until the the top left cells diagonally are either empty or of the same player
        # spaces_used_to_win is built, then it breaks
        spaces_used_to_win = 0
        for row in range(last_move_row + 1, -1):
            step = last_move_row - row
            if board.get_cell_value(row, (last_move_column - step)) == player or 0:
                chain += 1
            else:
                break

        if (chain + spaces_used_to_win) >= num_to_connect:
            move_guarantees_connect_4 = True

        return move_guarantees_connect_4

    # Prevents the player to scale up to the top of a column, if the move won't lead to winning state
    def prevent_vertical_upscale(self, board):
        prevention = False
        last_move = board.last_move
        last_move_row = last_move[0]
        last_move_column = last_move[1]

        chain = 0
        for row in range(board.height - 1, last_move_row):
            if board.get_cell_value(row, last_move_column) == board.self:
                chain += 1
            else:
                chain = 0

        if (chain + last_move_row) >= board.num_to_connect:
            prevention = True

        return prevention
