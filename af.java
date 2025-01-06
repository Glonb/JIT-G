class Calculator {
    public int add(int a, int b) {
        return a + b;
    }

    public int multiply(int a, int b) {
        return a * b;
    }

    public int calculate() {
        int sum = add(10, 20);
        if (sum > 25){
            int product = multiply(2, 3);
//             return sum + product;
            sum = sum + product;
        }else{
            System.out.println(sum);
        }
        return sum;
    }
}

public class Main {
    public static void main(String[] args) {
        Calculator calc = new Calculator();
        int result = calc.calculate();
        System.out.println(result);
    }
}
