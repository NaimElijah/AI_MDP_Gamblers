import matplotlib.pyplot as plt

from value_iteration import ValueIteration

if __name__ == "__main__":
    policy, v = ValueIteration().value_iteration_for_gamblers()

    print("Optimized Policy:")
    print(policy)
    print("")

    print("Optimized Value Function:")
    print(v)
    print("")

    # Plotting Final Policy (action stake) vs State (Capital)
    
    # x axis values
    x = range(100)
    # corresponding y axis values
    y = v[:100]

    # plotting the points
    plt.plot(x, y)

    # naming the x axis
    plt.xlabel("Capital")
    # naming the y axis
    plt.ylabel("Value Estimates")

    # giving a title to the graph
    plt.title("Final Policy (action stake) vs State (Capital)")

    # function to show the plot
    plt.show()
