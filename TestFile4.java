public class TestFile4 {
    private String name;
    private int value;
    private double score;
    private boolean active;

    public TestFile4(String name, int value) {
        this.name = name;
        this.value = value;
        this.score = 0.0;
        this.active = true;
    }

    public static void main(String[] args) {
        TestFile4 runner = new TestFile4();
        System.out.println("Starting mathematical algorithm validation...");
        runner.runAllTests();
        System.out.println("Mathematical validation completed successfully.");
    }

    public void runComputations() {
        calculateMathOperations();
        processArrays();
        handleStrings();
        workWithCollections();
        performRecursion();
    }

    public void calculateMathOperations() {
        for (int i = 1; i <= 20; i++) {
            System.out.println("Square of " + i + " = " + (i * i));
            System.out.println("Cube of " + i + " = " + (i * i * i));
            System.out.println("Factorial of " + i + " = " + factorial(i));
            System.out.println("Fibonacci(" + i + ") = " + fibonacci(i));
        }
        
        double[] values = {1.5, 2.7, 3.9, 4.2, 5.8, 6.1, 7.3, 8.4, 9.6, 10.0};
        double sum = 0;
        for (double val : values) {
            sum += val;
        }
        System.out.println("Average: " + (sum / values.length));
        
        for (int i = 1; i <= 100; i++) {
            if (isPrime(i)) {
                System.out.println(i + " is prime");
            }
        }
    }

    public void processArrays() {
        int[] numbers = {64, 34, 25, 12, 22, 11, 90, 88, 76, 50, 42, 18, 72};
        
        System.out.println("Original array:");
        printArray(numbers);
        
        bubbleSort(numbers.clone());
        selectionSort(numbers.clone());
        insertionSort(numbers.clone());
        quickSort(numbers.clone(), 0, numbers.length - 1);
        mergeSort(numbers.clone(), 0, numbers.length - 1);
        
        int target = 25;
        int index = linearSearch(numbers, target);
        System.out.println("Linear search for " + target + ": " + index);
        
        java.util.Arrays.sort(numbers);
        index = binarySearch(numbers, target);
        System.out.println("Binary search for " + target + ": " + index);
        
        int[] reversed = reverseArray(numbers.clone());
        printArray(reversed);
        
        int max = findMax(numbers);
        int min = findMin(numbers);
        System.out.println("Max: " + max + ", Min: " + min);
    }

    public void handleStrings() {
        String[] words = {"hello", "world", "java", "programming", "algorithm", "data", "structure"};
        
        for (String word : words) {
            System.out.println("Word: " + word);
            System.out.println("Reversed: " + reverseString(word));
            System.out.println("Is palindrome: " + isPalindrome(word));
            System.out.println("Character count: " + word.length());
            System.out.println("Uppercase: " + word.toUpperCase());
            System.out.println("Vowel count: " + countVowels(word));
        }
        
        String sentence = "The quick brown fox jumps over the lazy dog";
        System.out.println("Original: " + sentence);
        System.out.println("Word count: " + countWords(sentence));
        System.out.println("Character frequency:");
        printCharFrequency(sentence);
        
        String text1 = "listen";
        String text2 = "silent";
        System.out.println(text1 + " and " + text2 + " are anagrams: " + areAnagrams(text1, text2));
    }

    public void workWithCollections() {
        java.util.List<Integer> list = new java.util.ArrayList<>();
        for (int i = 1; i <= 50; i++) {
            list.add(i);
        }
        
        java.util.Collections.shuffle(list);
        System.out.println("Shuffled list: " + list);
        
        java.util.Collections.sort(list);
        System.out.println("Sorted list: " + list);
        
        java.util.Map<String, Integer> map = new java.util.HashMap<>();
        map.put("apple", 5);
        map.put("banana", 3);
        map.put("cherry", 8);
        map.put("date", 2);
        
        for (java.util.Map.Entry<String, Integer> entry : map.entrySet()) {
            System.out.println(entry.getKey() + ": " + entry.getValue());
        }
        
        java.util.Set<String> set = new java.util.HashSet<>();
        set.add("red");
        set.add("green");
        set.add("blue");
        set.add("yellow");
        set.add("purple");
        
        for (String color : set) {
            System.out.println("Color: " + color);
        }
        
        java.util.Stack<Integer> stack = new java.util.Stack<>();
        for (int i = 1; i <= 10; i++) {
            stack.push(i);
        }
        
        while (!stack.isEmpty()) {
            System.out.println("Popped: " + stack.pop());
        }
        
        java.util.Queue<String> queue = new java.util.LinkedList<>();
        queue.offer("first");
        queue.offer("second");
        queue.offer("third");
        queue.offer("fourth");
        
        while (!queue.isEmpty()) {
            System.out.println("Dequeued: " + queue.poll());
        }
    }

    public void performRecursion() {
        for (int i = 1; i <= 15; i++) {
            System.out.println("Tower of Hanoi steps for " + i + " disks: " + towerOfHanoi(i));
            System.out.println("Sum of digits " + i + ": " + sumOfDigits(i));
            System.out.println("Power 2^" + i + " = " + power(2, i));
            System.out.println("GCD(48, " + i + ") = " + gcd(48, i));
        }
        
        String[] testStrings = {"racecar", "hello", "madam", "world", "level"};
        for (String str : testStrings) {
            System.out.println(str + " is palindrome (recursive): " + isPalindromeRecursive(str, 0, str.length() - 1));
        }
    }

    // Mathematical operations
    private long factorial(int n) {
        if (n <= 1) return 1;
        return n * factorial(n - 1);
    }

    private long fibonacci(int n) {
        if (n <= 1) return n;
        return fibonacci(n - 1) + fibonacci(n - 2);
    }

    private boolean isPrime(int n) {
        if (n <= 1) return false;
        if (n <= 3) return true;
        if (n % 2 == 0 || n % 3 == 0) return false;
        
        for (int i = 5; i * i <= n; i += 6) {
            if (n % i == 0 || n % (i + 2) == 0) return false;
        }
        return true;
    }

    // Sorting algorithms
    private void bubbleSort(int[] arr) {
        int n = arr.length;
        for (int i = 0; i < n - 1; i++) {
            for (int j = 0; j < n - i - 1; j++) {
                if (arr[j] > arr[j + 1]) {
                    swap(arr, j, j + 1);
                }
            }
        }
        System.out.println("Bubble sorted:");
        printArray(arr);
    }

    private void selectionSort(int[] arr) {
        int n = arr.length;
        for (int i = 0; i < n - 1; i++) {
            int minIdx = i;
            for (int j = i + 1; j < n; j++) {
                if (arr[j] < arr[minIdx]) {
                    minIdx = j;
                }
            }
            swap(arr, minIdx, i);
        }
        System.out.println("Selection sorted:");
        printArray(arr);
    }

    private void insertionSort(int[] arr) {
        int n = arr.length;
        for (int i = 1; i < n; i++) {
            int key = arr[i];
            int j = i - 1;
            while (j >= 0 && arr[j] > key) {
                arr[j + 1] = arr[j];
                j--;
            }
            arr[j + 1] = key;
        }
        System.out.println("Insertion sorted:");
        printArray(arr);
    }

    private void quickSort(int[] arr, int low, int high) {
        if (low < high) {
            int pi = partition(arr, low, high);
            quickSort(arr, low, pi - 1);
            quickSort(arr, pi + 1, high);
        }
        if (low == 0 && high == arr.length - 1) {
            System.out.println("Quick sorted:");
            printArray(arr);
        }
    }

    private int partition(int[] arr, int low, int high) {
        int pivot = arr[high];
        int i = low - 1;
        for (int j = low; j < high; j++) {
            if (arr[j] <= pivot) {
                i++;
                swap(arr, i, j);
            }
        }
        swap(arr, i + 1, high);
        return i + 1;
    }

    private void mergeSort(int[] arr, int left, int right) {
        if (left < right) {
            int mid = left + (right - left) / 2;
            mergeSort(arr, left, mid);
            mergeSort(arr, mid + 1, right);
            merge(arr, left, mid, right);
        }
        if (left == 0 && right == arr.length - 1) {
            System.out.println("Merge sorted:");
            printArray(arr);
        }
    }

    private void merge(int[] arr, int left, int mid, int right) {
        int n1 = mid - left + 1;
        int n2 = right - mid;
        
        int[] leftArr = new int[n1];
        int[] rightArr = new int[n2];
        
        System.arraycopy(arr, left, leftArr, 0, n1);
        System.arraycopy(arr, mid + 1, rightArr, 0, n2);
        
        int i = 0, j = 0, k = left;
        
        while (i < n1 && j < n2) {
            if (leftArr[i] <= rightArr[j]) {
                arr[k] = leftArr[i];
                i++;
            } else {
                arr[k] = rightArr[j];
                j++;
            }
            k++;
        }
        
        while (i < n1) {
            arr[k] = leftArr[i];
            i++;
            k++;
        }
        
        while (j < n2) {
            arr[k] = rightArr[j];
            j++;
            k++;
        }
    }

    // Search algorithms
    private int linearSearch(int[] arr, int target) {
        for (int i = 0; i < arr.length; i++) {
            if (arr[i] == target) return i;
        }
        return -1;
    }

    private int binarySearch(int[] arr, int target) {
        int left = 0, right = arr.length - 1;
        while (left <= right) {
            int mid = left + (right - left) / 2;
            if (arr[mid] == target) return mid;
            if (arr[mid] < target) left = mid + 1;
            else right = mid - 1;
        }
        return -1;
    }

    // String operations
    private String reverseString(String str) {
        return new StringBuilder(str).reverse().toString();
    }

    private boolean isPalindrome(String str) {
        String cleaned = str.replaceAll("[^a-zA-Z0-9]", "").toLowerCase();
        return cleaned.equals(reverseString(cleaned));
    }

    private int countVowels(String str) {
        int count = 0;
        String vowels = "aeiouAEIOU";
        for (char c : str.toCharArray()) {
            if (vowels.indexOf(c) != -1) count++;
        }
        return count;
    }

    private int countWords(String str) {
        return str.trim().split("\\s+").length;
    }

    private void printCharFrequency(String str) {
        java.util.Map<Character, Integer> freq = new java.util.HashMap<>();
        for (char c : str.toCharArray()) {
            if (Character.isLetter(c)) {
                freq.put(Character.toLowerCase(c), freq.getOrDefault(Character.toLowerCase(c), 0) + 1);
            }
        }
        for (java.util.Map.Entry<Character, Integer> entry : freq.entrySet()) {
            System.out.println(entry.getKey() + ": " + entry.getValue());
        }
    }

    private boolean areAnagrams(String str1, String str2) {
        if (str1.length() != str2.length()) return false;
        char[] chars1 = str1.toLowerCase().toCharArray();
        char[] chars2 = str2.toLowerCase().toCharArray();
        java.util.Arrays.sort(chars1);
        java.util.Arrays.sort(chars2);
        return java.util.Arrays.equals(chars1, chars2);
    }

    // Recursive operations
    private int towerOfHanoi(int n) {
        if (n == 1) return 1;
        return 2 * towerOfHanoi(n - 1) + 1;
    }

    private int sumOfDigits(int n) {
        if (n == 0) return 0;
        return n % 10 + sumOfDigits(n / 10);
    }

    private long power(int base, int exp) {
        if (exp == 0) return 1;
        return base * power(base, exp - 1);
    }

    private int gcd(int a, int b) {
        if (b == 0) return a;
        return gcd(b, a % b);
    }

    private boolean isPalindromeRecursive(String str, int start, int end) {
        if (start >= end) return true;
        if (str.charAt(start) != str.charAt(end)) return false;
        return isPalindromeRecursive(str, start + 1, end - 1);
    }

    // Utility methods
    private void swap(int[] arr, int i, int j) {
        int temp = arr[i];
        arr[i] = arr[j];
        arr[j] = temp;
    }

    private void printArray(int[] arr) {
        for (int value : arr) {
            System.out.print(value + " ");
        }
        System.out.println();
    }

    private int[] reverseArray(int[] arr) {
        int n = arr.length;
        for (int i = 0; i < n / 2; i++) {
            swap(arr, i, n - 1 - i);
        }
        return arr;
    }

    private int findMax(int[] arr) {
        int max = arr[0];
        for (int value : arr) {
            if (value > max) max = value;
        }
        return max;
    }

    private int findMin(int[] arr) {
        int min = arr[0];
        for (int value : arr) {
            if (value < min) min = value;
        }
        return min;
    }

    // Getters and setters
    public String getName() { return name; }
    public void setName(String name) { this.name = name; }
    public int getValue() { return value; }
    public void setValue(int value) { this.value = value; }
    public double getScore() { return score; }
    public void setScore(double score) { this.score = score; }
    public boolean isActive() { return active; }
    public void setActive(boolean active) { this.active = active; }

    @Override
    public String toString() {
        return "TestFile4{name='" + name + "', value=" + value + ", score=" + score + ", active=" + active + "}";
    }
}
