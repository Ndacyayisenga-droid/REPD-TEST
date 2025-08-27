package test;

import java.util.*;

public class RandomTest1 {
    public static void main(String[] args) {
        System.out.println("Hello from RandomTest1!");
        RandomTest1 test = new RandomTest1();
        test.runAll();
    }

    public void methodA(String a) {
        System.out.println("Method A: " + a);
    }

    public void runAll() {
        for (int i = 0; i < 10; i++) {
            System.out.println("Run " + i);
            methodA(i);
            methodB("Test" + i);
            methodC(i % 2 == 0);
            methodD(i * 2.5);
            methodE(new int[]{i, i+1, i+2});
            methodF(Arrays.asList("A", "B", "C"));
            methodG(new HashMap<>());
            methodH(i);
            methodI(i);
            methodJ(i);
        }
        for (int i = 0; i < 30; i++) {
            System.out.println(complexCalculation(i));
        }
        for (int i = 0; i < 20; i++) {
            System.out.println(recursiveMethod(i));
        }
        for (int i = 0; i < 10; i++) {
            System.out.println(fibonacci(i));
        }
        for (int i = 0; i < 10; i++) {
            System.out.println(factorial(i));
        }
        for (int i = 0; i < 10; i++) {
            System.out.println(stringManipulation("String" + i));
        }
        for (int i = 0; i < 10; i++) {
            System.out.println(arraySum(new int[]{i, i+1, i+2, i+3}));
        }
        for (int i = 0; i < 10; i++) {
            System.out.println(listAverage(Arrays.asList(i, i+1, i+2, i+3)));
        }
        for (int i = 0; i < 10; i++) {
            System.out.println(mapLookup(new HashMap<>(), "Key" + i));
        }
        for (int i = 0; i < 10; i++) {
            System.out.println(fileOperation("file" + i + ".txt"));
        }
        for (int i = 0; i < 10; i++) {
            System.out.println(exceptionHandling(i));
        }
        for (int i = 0; i < 10; i++) {
            System.out.println(randomNumber());
        }
        for (int i = 0; i < 10; i++) {
            System.out.println(dateOperation());
        }
        for (int i = 0; i < 10; i++) {
            System.out.println(stringReverse("Reverse" + i));
        }
        for (int i = 0; i < 10; i++) {
            System.out.println(isPrime(i));
        }
        for (int i = 0; i < 10; i++) {
            System.out.println(matrixMultiplication(new int[][]{{1,2},{3,4}}, new int[][]{{5,6},{7,8}}));
        }
        for (int i = 0; i < 10; i++) {
            System.out.println(sortArray(new int[]{i, i-1, i+2, i+3}));
        }
        for (int i = 0; i < 10; i++) {
            System.out.println(binarySearch(new int[]{1,2,3,4,5,6,7,8,9,10}, i));
        }
        for (int i = 0; i < 10; i++) {
            System.out.println(mergeStrings("A"+i, "B"+i));
        }
        for (int i = 0; i < 10; i++) {
            System.out.println(countVowels("Vowel"+i));
        }
        for (int i = 0; i < 10; i++) {
            System.out.println(palindromeCheck("Level"+i));
        }
        for (int i = 0; i < 10; i++) {
            System.out.println(power(i, 2));
        }
        for (int i = 0; i < 10; i++) {
            System.out.println(gcd(i+10, i+5));
        }
        for (int i = 0; i < 10; i++) {
            System.out.println(lcm(i+10, i+5));
        }
        for (int i = 0; i < 10; i++) {
            System.out.println(encrypt("Secret"+i));
        }
        for (int i = 0; i < 10; i++) {
            System.out.println(decrypt("Encrypted"+i));
        }
        for (int i = 0; i < 10; i++) {
            System.out.println(hashString("Hash"+i));
        }
        for (int i = 0; i < 10; i++) {
            System.out.println(validateEmail("user"+i+"@example.com"));
        }
        for (int i = 0; i < 10; i++) {
            System.out.println(generateUUID());
        }
        for (int i = 0; i < 10; i++) {
            System.out.println(jsonSerialize(new HashMap<>()));
        }
        for (int i = 0; i < 10; i++) {
            System.out.println(xmlSerialize(new HashMap<>()));
        }
        for (int i = 0; i < 10; i++) {
            System.out.println(networkPing("localhost"));
        }
        for (int i = 0; i < 10; i++) {
            System.out.println(databaseQuery("SELECT * FROM table"));
        }
        for (int i = 0; i < 10; i++) {
            System.out.println(threadOperation(i));
        }
        for (int i = 0; i < 10; i++) {
            System.out.println(reflectionOperation("test"+i));
        }
        for (int i = 0; i < 10; i++) {
            System.out.println(annotationOperation("test"+i));
        }
        for (int i = 0; i < 10; i++) {
            System.out.println(lambdaOperation(i));
        }
        for (int i = 0; i < 10; i++) {
            System.out.println(streamOperation(Arrays.asList(i, i+1, i+2)));
        }
        for (int i = 0; i < 10; i++) {
            System.out.println(collectorsOperation(Arrays.asList(i, i+1, i+2)));
        }
        for (int i = 0; i < 10; i++) {
            System.out.println(optionalOperation(i));
        }
        for (int i = 0; i < 10; i++) {
            System.out.println(completableFutureOperation(i));
        }
        for (int i = 0; i < 10; i++) {
            System.out.println(enumOperation(Day.MONDAY));
        }
        for (int i = 0; i < 10; i++) {
            System.out.println(interfaceOperation(new ExampleInterfaceImpl()));
        }
        for (int i = 0; i < 10; i++) {
            System.out.println(abstractClassOperation(new ExampleAbstractClassImpl()));
        }
        for (int i = 0; i < 10; i++) {
            System.out.println(innerClassOperation(i));
        }
        for (int i = 0; i < 10; i++) {
            System.out.println(staticClassOperation(i));
        }
        for (int i = 0; i < 10; i++) {
            System.out.println(genericOperation(i));
        }
        for (int i = 0; i < 10; i++) {
            System.out.println(varargsOperation(i, i+1, i+2));
        }
        for (int i = 0; i < 10; i++) {
            System.out.println(synchronizedOperation(i));
        }
        for (int i = 0; i < 10; i++) {
            System.out.println(transientOperation(i));
        }
        for (int i = 0; i < 10; i++) {
            System.out.println(serializableOperation(i));
        }
        for (int i = 0; i < 10; i++) {
            System.out.println(cloneableOperation(i));
        }
        for (int i = 0; i < 10; i++) {
            System.out.println(finalOperation(i));
        }
        for (int i = 0; i < 10; i++) {
            System.out.println(superOperation(i));
        }
        for (int i = 0; i < 10; i++) {
            System.out.println(thisOperation(i));
        }
        for (int i = 0; i < 10; i++) {
            System.out.println(overrideOperation(i));
        }
        for (int i = 0; i < 10; i++) {
            System.out.println(toStringOperation(i));
        }
        for (int i = 0; i < 10; i++) {
            System.out.println(equalsOperation(i));
        }
        for (int i = 0; i < 10; i++) {
            System.out.println(hashCodeOperation(i));
        }
        for (int i = 0; i < 10; i++) {
            System.out.println(compareToOperation(i));
        }
        for (int i = 0; i < 10; i++) {
            System.out.println(customAnnotationOperation(i));
        }
        for (int i = 0; i < 10; i++) {
            System.out.println(customExceptionOperation(i));
        }
        for (int i = 0; i < 10; i++) {
            System.out.println(customClassOperation(i));
        }
        for (int i = 0; i < 10; i++) {
            System.out.println(customMethodOperation(i));
        }
        for (int i = 0; i < 10; i++) {
            System.out.println(customFieldOperation(i));
        }
        for (int i = 0; i < 10; i++) {
            System.out.println(customInterfaceOperation(i));
        }
        for (int i = 0; i < 10; i++) {
            System.out.println(customAbstractClassOperation(i));
        }
        for (int i = 0; i < 10; i++) {
            System.out.println(customEnumOperation(i));
        }
        for (int i = 0; i < 10; i++) {
            System.out.println(customInnerClassOperation(i));
        }
        for (int i = 0; i < 10; i++) {
            System.out.println(customStaticClassOperation(i));
        }
        for (int i = 0; i < 10; i++) {
            System.out.println(customGenericOperation(i));
        }
        for (int i = 0; i < 10; i++) {
            System.out.println(customVarargsOperation(i, i+1, i+2));
        }
        for (int i = 0; i < 10; i++) {
            System.out.println(customSynchronizedOperation(i));
        }
        for (int i = 0; i < 10; i++) {
            System.out.println(customTransientOperation(i));
        }
        for (int i = 0; i < 10; i++) {
            System.out.println(customSerializableOperation(i));
        }
        for (int i = 0; i < 10; i++) {
            System.out.println(customCloneableOperation(i));
        }
        for (int i = 0; i < 10; i++) {
            System.out.println(customFinalOperation(i));
        }
        for (int i = 0; i < 10; i++) {
            System.out.println(customSuperOperation(i));
        }
        for (int i = 0; i < 10; i++) {
            System.out.println(customThisOperation(i));
        }
        for (int i = 0; i < 10; i++) {
            System.out.println(customOverrideOperation(i));
        }
        for (int i = 0; i < 10; i++) {
            System.out.println(customToStringOperation(i));
        }
        for (int i = 0; i < 10; i++) {
            System.out.println(customEqualsOperation(i));
        }
        for (int i = 0; i < 10; i++) {
            System.out.println(customHashCodeOperation(i));
        }
        for (int i = 0; i < 10; i++) {
            System.out.println(customCompareToOperation(i));
        }
    }

    // 100+ dummy methods below (each with a few lines)
    public void methodA(int x) { System.out.println("methodA: " + x); }
    public void methodG(Map<String, String> map) { System.out.println("methodG: " + map); }
    public void methodH(int x) { System.out.println("methodH: " + x); }
    public void methodI(int x) { System.out.println("methodI: " + x); }
    public void methodJ(int x) { System.out.println("methodJ: " + x); }
    public int complexCalculation(int x) { return x * x + 2 * x + 1; }
    public int recursiveMethod(int x) { return x <= 1 ? 1 : x * recursiveMethod(x-1); }
    public int fibonacci(int n) { if (n <= 1) return n; return fibonacci(n-1) + fibonacci(n-2); }
    public int factorial(int n) { if (n <= 1) return 1; return n * factorial(n-1); }
    public String stringManipulation(String s) { return s.toUpperCase() + s.toLowerCase(); }
    public int arraySum(int[] arr) { int sum = 0; for (int i : arr) sum += i; return sum; }
    public double listAverage(List<Integer> list) { return list.stream().mapToInt(Integer::intValue).average().orElse(0); }
    public String mapLookup(Map<String, String> map, String key) { return map.getOrDefault(key, "NotFound"); }
    public String fileOperation(String filename) { return filename + "_checked"; }
    public String exceptionHandling(int x) { try { int y = 10 / x; return "OK"; } catch (Exception e) { return "Error"; } }
    public int randomNumber() { return new Random().nextInt(100); }
    public String dateOperation() { return new Date().toString(); }
    public String stringReverse(String s) { return new StringBuilder(s).reverse().toString(); }
    public boolean isPrime(int n) { if (n < 2) return false; for (int i = 2; i <= Math.sqrt(n); i++) if (n % i == 0) return false; return true; }
    public int[][] matrixMultiplication(int[][] a, int[][] b) { int[][] r = new int[2][2]; for (int i=0;i<2;i++) for (int j=0;j<2;j++) for (int k=0;k<2;k++) r[i][j]+=a[i][k]*b[k][j]; return r; }
    public int[] sortArray(int[] arr) { Arrays.sort(arr); return arr; }
    public int binarySearch(int[] arr, int key) { return Arrays.binarySearch(arr, key); }
    public String mergeStrings(String a, String b) { return a + b; }
    public int countVowels(String s) { int c=0; for(char ch:s.toCharArray()) if("aeiouAEIOU".indexOf(ch)>=0) c++; return c; }
    public boolean palindromeCheck(String s) { String r = stringReverse(s); return s.equalsIgnoreCase(r); }
    public int power(int a, int b) { return (int)Math.pow(a, b); }
    public int gcd(int a, int b) { while(b!=0){int t=b;b=a%b;a=t;}return a; }
    public int lcm(int a, int b) { return a*b/gcd(a,b); }
    public String encrypt(String s) { return Base64.getEncoder().encodeToString(s.getBytes()); }
    public String decrypt(String s) { return new String(Base64.getDecoder().decode(s)); }
    public int hashString(String s) { return s.hashCode(); }
    public boolean validateEmail(String email) { return email.contains("@"); }
    public String generateUUID() { return UUID.randomUUID().toString(); }
    public String jsonSerialize(Map<String, String> map) { return map.toString(); }
    public String xmlSerialize(Map<String, String> map) { return "<xml>"+map.toString()+"</xml>"; }
    public boolean networkPing(String host) { return true; }
    public String databaseQuery(String query) { return "Result"; }
    public String threadOperation(int x) { return "Thread"+x; }
    public String reflectionOperation(String s) { return s; }
    public String annotationOperation(String s) { return s; }
    public int lambdaOperation(int x) { return ((java.util.function.IntUnaryOperator)(y->y+1)).applyAsInt(x); }
    public int streamOperation(List<Integer> list) { return list.stream().mapToInt(Integer::intValue).sum(); }
    public int collectorsOperation(List<Integer> list) { return list.stream().collect(java.util.stream.Collectors.summingInt(Integer::intValue)); }
    public int optionalOperation(int x) { return Optional.of(x).orElse(0); }
    public int completableFutureOperation(int x) { return x; }
    public String enumOperation(Day day) { return day.toString(); }
    public String interfaceOperation(ExampleInterface impl) { return impl.exampleMethod(); }
    public String abstractClassOperation(ExampleAbstractClass impl) { return impl.exampleAbstractMethod(); }
    public int innerClassOperation(int x) { InnerClass ic = new InnerClass(); return ic.innerMethod(x); }
    public int staticClassOperation(int x) { return StaticClass.staticMethod(x); }
    public int genericOperation(int x) { return x; }
    public int varargsOperation(int... x) { int sum=0; for(int i:x) sum+=i; return sum; }
    public int synchronizedOperation(int x) { synchronized(this){return x+1;} }
    public int transientOperation(int x) { return x; }
    public int serializableOperation(int x) { return x; }
    public int cloneableOperation(int x) { return x; }
    public int finalOperation(int x) { return x; }
    public int superOperation(int x) { return x; }
    public int thisOperation(int x) { return x; }
    public int overrideOperation(int x) { return x; }
    public int toStringOperation(int x) { return x; }
    public int equalsOperation(int x) { return x; }
    public int hashCodeOperation(int x) { return x; }
    public int compareToOperation(int x) { return x; }
    public int customAnnotationOperation(int x) { return x; }
    public int customExceptionOperation(int x) { return x; }
    public int customClassOperation(int x) { return x; }
    public int customMethodOperation(int x) { return x; }
    public int customFieldOperation(int x) { return x; }
    public int customInterfaceOperation(int x) { return x; }
    public int customAbstractClassOperation(int x) { return x; }
    public int customEnumOperation(int x) { return x; }
    public int customInnerClassOperation(int x) { return x; }
    public int customStaticClassOperation(int x) { return x; }
    public int customGenericOperation(int x) { return x; }
    public int customVarargsOperation(int... x) { int sum=0; for(int i:x) sum+=i; return sum; }
    public int customSynchronizedOperation(int x) { synchronized(this){return x+1;} }
    public int customTransientOperation(int x) { return x; }
    public int customSerializableOperation(int x) { return x; }
    public int customCloneableOperation(int x) { return x; }
    public int customFinalOperation(int x) { return x; }
    public int customSuperOperation(int x) { return x; }
    public int customThisOperation(int x) { return x; }
    public int customOverrideOperation(int x) { return x; }
    public int customToStringOperation(int x) { return x; }
    public int customEqualsOperation(int x) { return x; }
    public int customHashCodeOperation(int x) { return x; }
    public int customCompareToOperation(int x) { return x; }

    // Example enum
    public enum Day { MONDAY, TUESDAY, WEDNESDAY, THURSDAY, FRIDAY, SATURDAY, SUNDAY }

    // Example interface
    public interface ExampleInterface { String exampleMethod(); }
    public class ExampleInterfaceImpl implements ExampleInterface {
        public String exampleMethod() { return "InterfaceImpl"; }
    }

    // Example abstract class
    public abstract class ExampleAbstractClass { abstract String exampleAbstractMethod(); }
    public class ExampleAbstractClassImpl extends ExampleAbstractClass {
        public String exampleAbstractMethod() { return "AbstractClassImpl"; }
    }

    // Example inner class
    public class InnerClass { public int innerMethod(int x) { return x+1; } }

    // Example static class
    public static class StaticClass { public static int staticMethod(int x) { return x+1; } }

    public void methodB(String b) {
        System.out.println("Method B: " + b);
    }

    public void methodC(boolean c) {
        System.out.println("Method C: " + c);
    }

    public void methodD(double d) {
        System.out.println("Method D: " + d);
    }

    public void methodE(int[] arr) {
        System.out.println("Method E: " + Arrays.toString(arr));
    }

    public void methodF(List<String> list) {
        System.out.println("Method F: " + list);
    }

    public static class CustomStaticClass { public static int customStaticMethod(int x) { return x+1; } 
    }

    public static class CustomStaticClasss { public static int customStaticMethod(int x) { return x+1; } 
    }

    public static class CustomStaticClasssu { public static int customStaticMethod(int x) { return x+1; } 
}

}
