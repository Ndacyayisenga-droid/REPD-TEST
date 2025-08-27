public class TestFile6 {
    public static void main(String[] args) {
        TestFile6 test = new TestFile6();
        test.runTests();
    }

    public void runTests() {
        testBasicOperations();
        testDataProcessing();
        testAlgorithms();
    }

    public void testBasicOperations() {
        for (int i = 0; i < 50; i++) {
            System.out.println("Processing item " + i);
            calculateValue(i);
            processString("test" + i);
            handleArray(new int[]{i, i+1, i+2});
        }
    }

    public void testDataProcessing() {
        java.util.List<String> data = new java.util.ArrayList<>();
        for (int i = 0; i < 100; i++) {
            data.add("item" + i);
        }
        
        java.util.Map<String, Integer> counts = new java.util.HashMap<>();
        for (String item : data) {
            counts.put(item, item.length());
        }
        
        java.util.Set<String> uniqueItems = new java.util.HashSet<>(data);
        System.out.println("Unique items: " + uniqueItems.size());
    }

    public void testAlgorithms() {
        int[] array = {5, 2, 8, 1, 9, 3, 7, 4, 6};
        sortArray(array);
        searchInArray(array, 5);
        reverseArray(array);
        
        String text = "hello world java programming";
        processText(text);
        findPatterns(text);
    }

    private void calculateValue(int input) {
        int result = input * input + input + 1;
        System.out.println("Calculated: " + result);
    }

    private void processString(String str) {
        String reversed = new StringBuilder(str).reverse().toString();
        String uppercase = str.toUpperCase();
        System.out.println("Processed: " + reversed + ", " + uppercase);
    }

    private void handleArray(int[] arr) {
        int sum = 0;
        for (int value : arr) {
            sum += value;
        }
        System.out.println("Array sum: " + sum);
    }

    private void sortArray(int[] arr) {
        java.util.Arrays.sort(arr);
        System.out.println("Sorted: " + java.util.Arrays.toString(arr));
    }

    private void searchInArray(int[] arr, int target) {
        int index = java.util.Arrays.binarySearch(arr, target);
        System.out.println("Found " + target + " at index: " + index);
    }

    private void reverseArray(int[] arr) {
        for (int i = 0; i < arr.length / 2; i++) {
            int temp = arr[i];
            arr[i] = arr[arr.length - 1 - i];
            arr[arr.length - 1 - i] = temp;
        }
        System.out.println("Reversed: " + java.util.Arrays.toString(arr));
    }

    private void processText(String text) {
        String[] words = text.split(" ");
        System.out.println("Word count: " + words.length);
        
        for (String word : words) {
            System.out.println("Word: " + word + ", Length: " + word.length());
        }
    }

    private void findPatterns(String text) {
        int vowelCount = 0;
        for (char c : text.toCharArray()) {
            if ("aeiouAEIOU".indexOf(c) != -1) {
                vowelCount++;
            }
        }
        System.out.println("Vowels found: " + vowelCount);
    }
}
