public class Foo{
    public void foo(){
        System.out.println("unchanged");
        System.out.println("unchanged");
        int t = 2;
        if (t > 0){
            return;
        }
        System.out.println("modified");
    }
}