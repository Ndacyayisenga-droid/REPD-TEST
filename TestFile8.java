public class TestFile8 {
    public static void main(String[] args) {
        TestFile8 test = new TestFile8();
        test.runAllOperations();
    }

    public void runAllOperations() {
        testStringOperations();
        testArrayManipulation();
        testObjectOrientedFeatures();
        testExceptionHandling();
    }

    public void testStringOperations() {
        String[] testStrings = {
            "Hello World", "Java Programming", "Data Structures",
            "Algorithms", "Software Engineering", "Computer Science"
        };
        
        for (String str : testStrings) {
            processString(str);
            analyzeString(str);
            transformString(str);
        }
        
        testStringComparison();
        testStringBuilding();
    }

    public void testArrayManipulation() {
        int[][] matrix = {
            {1, 2, 3, 4},
            {5, 6, 7, 8},
            {9, 10, 11, 12},
            {13, 14, 15, 16}
        };
        
        printMatrix(matrix);
        transposeMatrix(matrix);
        findMatrixSum(matrix);
        
        int[] array1D = {64, 34, 25, 12, 22, 11, 90};
        performSorting(array1D);
        performSearching(array1D);
    }

    public void testObjectOrientedFeatures() {
        Person[] people = {
            new Person("Alice", 25, "Engineer"),
            new Person("Bob", 30, "Manager"),
            new Person("Charlie", 35, "Designer"),
            new Person("Diana", 28, "Analyst")
        };
        
        for (Person person : people) {
            person.displayInfo();
            person.work();
            person.celebrate();
        }
        
        testInheritance();
        testPolymorphism();
    }

    public void testExceptionHandling() {
        int[] numbers = {10, 5, 0, 15, 20};
        
        for (int i = 0; i < numbers.length; i++) {
            try {
                int result = divide(100, numbers[i]);
                System.out.println("100 / " + numbers[i] + " = " + result);
            } catch (ArithmeticException e) {
                System.err.println("Division by zero error: " + e.getMessage());
            }
        }
        
        testArrayAccess();
        testNullPointer();
        testCustomException();
    }

    private void processString(String str) {
        System.out.println("Processing: " + str);
        System.out.println("Length: " + str.length());
        System.out.println("Uppercase: " + str.toUpperCase());
        System.out.println("Lowercase: " + str.toLowerCase());
        System.out.println("Reversed: " + reverseString(str));
    }

    private void analyzeString(String str) {
        int vowels = countVowels(str);
        int consonants = countConsonants(str);
        int words = countWords(str);
        
        System.out.println("Analysis of '" + str + "':");
        System.out.println("Vowels: " + vowels);
        System.out.println("Consonants: " + consonants);
        System.out.println("Words: " + words);
    }

    private void transformString(String str) {
        String noSpaces = str.replaceAll("\\s+", "");
        String firstLetters = getFirstLetters(str);
        String alternating = alternateCase(str);
        
        System.out.println("Transformed '" + str + "':");
        System.out.println("No spaces: " + noSpaces);
        System.out.println("First letters: " + firstLetters);
        System.out.println("Alternating case: " + alternating);
    }

    private void testStringComparison() {
        String str1 = "hello";
        String str2 = "world";
        String str3 = "hello";
        
        System.out.println("String comparison:");
        System.out.println(str1 + " equals " + str2 + ": " + str1.equals(str2));
        System.out.println(str1 + " equals " + str3 + ": " + str1.equals(str3));
        System.out.println(str1 + " compare to " + str2 + ": " + str1.compareTo(str2));
    }

    private void testStringBuilding() {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < 10; i++) {
            sb.append("Item ").append(i).append(" ");
        }
        
        String result = sb.toString();
        System.out.println("Built string: " + result);
        
        StringBuffer sbuf = new StringBuffer();
        sbuf.append("Thread-safe string building");
        System.out.println("StringBuffer result: " + sbuf.toString());
    }

    private void printMatrix(int[][] matrix) {
        System.out.println("Matrix:");
        for (int[] row : matrix) {
            for (int value : row) {
                System.out.printf("%4d", value);
            }
            System.out.println();
        }
    }

    private void transposeMatrix(int[][] matrix) {
        System.out.println("Transposed matrix:");
        for (int j = 0; j < matrix[0].length; j++) {
            for (int i = 0; i < matrix.length; i++) {
                System.out.printf("%4d", matrix[i][j]);
            }
            System.out.println();
        }
    }

    private void findMatrixSum(int[][] matrix) {
        int sum = 0;
        for (int[] row : matrix) {
            for (int value : row) {
                sum += value;
            }
        }
        System.out.println("Matrix sum: " + sum);
    }

    private void performSorting(int[] array) {
        System.out.println("Original array: " + java.util.Arrays.toString(array));
        
        int[] bubbleSorted = array.clone();
        bubbleSort(bubbleSorted);
        System.out.println("Bubble sorted: " + java.util.Arrays.toString(bubbleSorted));
        
        int[] selectionSorted = array.clone();
        selectionSort(selectionSorted);
        System.out.println("Selection sorted: " + java.util.Arrays.toString(selectionSorted));
        
        int[] insertionSorted = array.clone();
        insertionSort(insertionSorted);
        System.out.println("Insertion sorted: " + java.util.Arrays.toString(insertionSorted));
    }

    private void performSearching(int[] array) {
        java.util.Arrays.sort(array);
        
        int target = 25;
        int linearResult = linearSearch(array, target);
        int binaryResult = binarySearch(array, target);
        
        System.out.println("Searching for " + target + ":");
        System.out.println("Linear search result: " + linearResult);
        System.out.println("Binary search result: " + binaryResult);
    }

    private void testInheritance() {
        Student student = new Student("John", 20, "Computer Science", "S12345");
        student.displayInfo();
        student.study();
        student.takeExam();
        
        Teacher teacher = new Teacher("Dr. Smith", 45, "Mathematics", "T001");
        teacher.displayInfo();
        teacher.teach();
        teacher.gradePapers();
    }

    private void testPolymorphism() {
        Person[] people = {
            new Student("Alice", 19, "Physics", "S67890"),
            new Teacher("Prof. Johnson", 50, "Chemistry", "T002"),
            new Person("Bob", 25, "Engineer")
        };
        
        for (Person person : people) {
            person.displayInfo();
            person.work(); // Polymorphic method call
        }
    }

    private void testArrayAccess() {
        int[] smallArray = {1, 2, 3, 4, 5};
        
        try {
            int value = smallArray[10]; // This will throw ArrayIndexOutOfBoundsException
            System.out.println("Value: " + value);
        } catch (ArrayIndexOutOfBoundsException e) {
            System.err.println("Array index out of bounds: " + e.getMessage());
        }
    }

    private void testNullPointer() {
        String nullString = null;
        
        try {
            int length = nullString.length(); // This will throw NullPointerException
            System.out.println("Length: " + length);
        } catch (NullPointerException e) {
            System.err.println("Null pointer exception: " + e.getMessage());
        }
    }

    private void testCustomException() {
        try {
            validateAge(-5);
        } catch (InvalidAgeException e) {
            System.err.println("Custom exception: " + e.getMessage());
        }
        
        try {
            validateAge(25);
            System.out.println("Age validation passed");
        } catch (InvalidAgeException e) {
            System.err.println("Custom exception: " + e.getMessage());
        }
    }

    // Helper methods
    private String reverseString(String str) {
        return new StringBuilder(str).reverse().toString();
    }

    private int countVowels(String str) {
        int count = 0;
        String vowels = "aeiouAEIOU";
        for (char c : str.toCharArray()) {
            if (vowels.indexOf(c) != -1) {
                count++;
            }
        }
        return count;
    }

    private int countConsonants(String str) {
        int count = 0;
        for (char c : str.toCharArray()) {
            if (Character.isLetter(c) && "aeiouAEIOU".indexOf(c) == -1) {
                count++;
            }
        }
        return count;
    }

    private int countWords(String str) {
        return str.trim().split("\\s+").length;
    }

    private String getFirstLetters(String str) {
        StringBuilder result = new StringBuilder();
        String[] words = str.split("\\s+");
        for (String word : words) {
            if (!word.isEmpty()) {
                result.append(word.charAt(0));
            }
        }
        return result.toString();
    }

    private String alternateCase(String str) {
        StringBuilder result = new StringBuilder();
        for (int i = 0; i < str.length(); i++) {
            char c = str.charAt(i);
            if (i % 2 == 0) {
                result.append(Character.toLowerCase(c));
            } else {
                result.append(Character.toUpperCase(c));
            }
        }
        return result.toString();
    }

    private int divide(int a, int b) throws ArithmeticException {
        if (b == 0) {
            throw new ArithmeticException("Division by zero");
        }
        return a / b;
    }

    private void validateAge(int age) throws InvalidAgeException {
        if (age < 0) {
            throw new InvalidAgeException("Age cannot be negative: " + age);
        }
        if (age > 150) {
            throw new InvalidAgeException("Age is too high: " + age);
        }
    }

    // Sorting algorithms
    private void bubbleSort(int[] arr) {
        int n = arr.length;
        for (int i = 0; i < n - 1; i++) {
            for (int j = 0; j < n - i - 1; j++) {
                if (arr[j] > arr[j + 1]) {
                    int temp = arr[j];
                    arr[j] = arr[j + 1];
                    arr[j + 1] = temp;
                }
            }
        }
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
            int temp = arr[minIdx];
            arr[minIdx] = arr[i];
            arr[i] = temp;
        }
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
    }

    // Search algorithms
    private int linearSearch(int[] arr, int target) {
        for (int i = 0; i < arr.length; i++) {
            if (arr[i] == target) {
                return i;
            }
        }
        return -1;
    }

    private int binarySearch(int[] arr, int target) {
        int left = 0, right = arr.length - 1;
        
        while (left <= right) {
            int mid = left + (right - left) / 2;
            
            if (arr[mid] == target) {
                return mid;
            }
            
            if (arr[mid] < target) {
                left = mid + 1;
            } else {
                right = mid - 1;
            }
        }
        
        return -1;
    }

    // Classes for OOP testing
    static class Person {
        protected String name;
        protected int age;
        protected String profession;
        
        public Person(String name, int age, String profession) {
            this.name = name;
            this.age = age;
            this.profession = profession;
        }
        
        public void displayInfo() {
            System.out.println("Name: " + name + ", Age: " + age + ", Profession: " + profession);
        }
        
        public void work() {
            System.out.println(name + " is working as a " + profession);
        }
        
        public void celebrate() {
            System.out.println(name + " is celebrating!");
        }
    }
    
    static class Student extends Person {
        private String studentId;
        
        public Student(String name, int age, String major, String studentId) {
            super(name, age, major);
            this.studentId = studentId;
        }
        
        @Override
        public void work() {
            System.out.println(name + " is studying " + profession);
        }
        
        public void study() {
            System.out.println(name + " is studying hard for " + profession);
        }
        
        public void takeExam() {
            System.out.println(name + " is taking an exam in " + profession);
        }
    }
    
    static class Teacher extends Person {
        private String employeeId;
        
        public Teacher(String name, int age, String subject, String employeeId) {
            super(name, age, subject);
            this.employeeId = employeeId;
        }
        
        @Override
        public void work() {
            System.out.println(name + " is teaching " + profession);
        }
        
        public void teach() {
            System.out.println(name + " is teaching a " + profession + " class");
        }
        
        public void gradePapers() {
            System.out.println(name + " is grading " + profession + " papers");
        }
    }
    
    // Custom exception
    static class InvalidAgeException extends Exception {
        public InvalidAgeException(String message) {
            super(message);
        }
    }
}
