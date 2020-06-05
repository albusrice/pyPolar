
prices = [1.0, 1.2, 1.5]
quantity = [0] * len(prices)
order_price = [0] * len(prices)


def print_options(item, choice):
    num_items = int(input("Please enter how many {} you want: ".format(item)))
    price = num_items * prices[choice - 1]
    print("You have ordered {} {}. Price is ${:.2f}\n\n ".format(num_items, item, price))
    return num_items,  price


def print_order(item, index):
    print("You have ordered {} {}. Price is ${:.2f}\n\n ".format(quantity[index],
                                                                 item,
                                                                 order_price[index]))

def burger():
    choice = 1

    while choice < 4:
        print("1. Beef burger $1.00")
        print("2. Cheese Burger $1.20")
        print("3. Chicken Burger $1.50")
        choice = int(input("Please enter your choice or '4' to go back to main menu : "))

        if choice == 1:
            quantity[choice - 1], order_price[choice - 1] = print_options("Beef Burger", choice)
        elif choice == 2:
            quantity[choice - 1], order_price[choice - 1] = print_options("Cheese Burger", choice)
        elif choice == 3:
            quantity[choice - 1], order_price[choice - 1] = print_options("Chicken Burger", choice)

    if any(quantity):
        return True
    else:
        return False

main_choice = 1
burger_status = False

print("Welcome to McDowell Burger Restaurant!")
while main_choice != 4:
    main_choice = int(input("Please select '1' for Burgers, '2' for Side Orders, '3' for Drinks  '4' for check out : "))
    if main_choice == 1:
        burger_status = burger()
    elif main_choice == 4:
        if burger_status == 1:
            for i in range(len(order_price)):

                if order_price[i] > 0.0:
                    if i == 0:
                        print_order("Beef Burger", i)
                    if i == 1:
                        print_order("Cheese Burger", i)
                    if i == 2:
                        print_order("Chicken Burger", i)





