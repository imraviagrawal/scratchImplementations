import numpy as np

def greedy_decoder(data):
    return [np.argmax(d) for d in data]

def beam_search_decoder(data, k):
    sequences = [(list(), 0.0)] # initialize

    # iterate through each candidates
    for d in data:
        all_candidates = []

        # iterate through each prespective candidates
        for i in range(len(sequences)):
            sequence, score = sequences[i]

            # iterate throught each values in the current row to be added.
            # all each previous candidates and add new candidate
            for j in range(len(d)):
                all_candidates.append((sequence+[j], score-np.log(d[j])))

        # select the best candidates
        all_candidates = sorted(all_candidates, key=lambda x: x[1])
        # select top candidates
        sequences = all_candidates[:k]
    return sequences



if __name__ == "__main__":

    # initialize dataset
    data = [[0.1, 0.2, 0.3, 0.4, 0.5],
            [0.5, 0.4, 0.3, 0.2, 0.1],
            [0.1, 0.2, 0.3, 0.4, 0.5],
            [0.5, 0.4, 0.3, 0.2, 0.1],
            [0.1, 0.2, 0.3, 0.4, 0.5],
            [0.5, 0.4, 0.3, 0.2, 0.1],
            [0.1, 0.2, 0.3, 0.4, 0.5],
            [0.5, 0.4, 0.3, 0.2, 0.1],
            [0.1, 0.2, 0.3, 0.4, 0.5],
            [0.5, 0.4, 0.3, 0.2, 0.1]]
    data = np.array(data)

    print("greedy decoding")
    print(greedy_decoder(data))

    beam_size=3
    print(f"Beam Seach Decoding with beam size {beam_size}")
    print(beam_search_decoder(data, beam_size))