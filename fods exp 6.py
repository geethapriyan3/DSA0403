# Item prices
prices = [100, 50, 200]

# Quantities purchased
quantities = [2, 3, 1]

# Discount and tax rates
discount_rate = 10   # 10%
tax_rate = 5         # 5%

# Calculate subtotal
subtotal = 0

for price, quantity in zip(prices, quantities):
    subtotal = subtotal + (price * quantity)

# Calculate discount
discount = subtotal * (discount_rate / 100)

# Price after discount
price_after_discount = subtotal - discount

# Calculate tax
tax = price_after_discount * (tax_rate / 100)

# Calculate final total
total_cost = price_after_discount + tax

# Display results
print("Subtotal:", subtotal)
print("Discount:", discount)
print("Price after discount:", price_after_discount)
print("Tax:", tax)
print("Total cost:", total_cost)
