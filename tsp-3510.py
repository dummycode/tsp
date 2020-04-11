import sys

def main(input_coords, output_tour, time):
    """
    Main driver for the program, kicks of search
    """
    print("Program started ({}, {}, {})".format(input_coords, output_tour, time))

if __name__ == "__main__":
    # Quit if invalid arguments
    if len(sys.argv) != 4:
        print("Invalid arguments!")
        quit(1)
    
    _, input_coords, output_tour, time = sys.argv

    main(input_coords, output_tour, time)

