class Calculator {
    public int add(int a, int b) {
        return a + b;
    }

    public int multiply(int a, int b) {
        return a * b;
    }

    public int calculate() {
        int sum = add(10, 20);
        int product = multiply(2, 3);
        return sum + product;
    }
}

public class Main {
    public static void main(String[] args) {
        Calculator calc = new Calculator();
        int result = calc.calculate();
        System.out.println(result);
    }
}
