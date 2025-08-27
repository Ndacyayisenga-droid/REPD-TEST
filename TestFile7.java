public class TestFile7 {
    public static void main(String[] args) {
        TestFile7 test = new TestFile7();
        test.executeTests();
    }

    public void executeTests() {
        performMathOperations();
        handleCollections();
        processFiles();
        testConcurrency();
    }

    public void performMathOperations() {
        for (int i = 1; i <= 30; i++) {
            int factorial = calculateFactorial(i);
            long fibonacci = calculateFibonacci(i);
            boolean isPrime = checkPrime(i);
            System.out.println("Number: " + i + ", Factorial: " + factorial + 
                             ", Fibonacci: " + fibonacci + ", Prime: " + isPrime);
        }
        
        double[] values = {1.1, 2.2, 3.3, 4.4, 5.5};
        double average = calculateAverage(values);
        double max = findMaximum(values);
        double min = findMinimum(values);
        System.out.println("Average: " + average + ", Max: " + max + ", Min: " + min);
    }

    public void handleCollections() {
        java.util.List<Integer> numbers = new java.util.ArrayList<>();
        for (int i = 1; i <= 50; i++) {
            numbers.add(i);
        }
        
        java.util.Collections.shuffle(numbers);
        java.util.Collections.sort(numbers);
        
        java.util.Map<String, Object> dataMap = new java.util.HashMap<>();
        dataMap.put("count", numbers.size());
        dataMap.put("first", numbers.get(0));
        dataMap.put("last", numbers.get(numbers.size() - 1));
        
        java.util.Set<Integer> evenNumbers = new java.util.TreeSet<>();
        for (int num : numbers) {
            if (num % 2 == 0) {
                evenNumbers.add(num);
            }
        }
        
        System.out.println("Even numbers: " + evenNumbers);
        System.out.println("Data map: " + dataMap);
    }

    public void processFiles() {
        try {
            java.nio.file.Path tempFile = java.nio.file.Files.createTempFile("test7", ".txt");
            
            java.util.List<String> lines = java.util.Arrays.asList(
                "This is line 1",
                "This is line 2",
                "This is line 3",
                "Processing file data",
                "Testing file operations"
            );
            
            java.nio.file.Files.write(tempFile, lines);
            
            java.util.List<String> readLines = java.nio.file.Files.readAllLines(tempFile);
            for (String line : readLines) {
                System.out.println("Read: " + line);
            }
            
            long fileSize = java.nio.file.Files.size(tempFile);
            System.out.println("File size: " + fileSize + " bytes");
            
            java.nio.file.Files.deleteIfExists(tempFile);
            
        } catch (Exception e) {
            System.err.println("File processing error: " + e.getMessage());
        }
    }

    public void testConcurrency() {
        java.util.concurrent.ExecutorService executor = java.util.concurrent.Executors.newFixedThreadPool(3);
        
        for (int i = 0; i < 5; i++) {
            final int taskId = i;
            executor.submit(() -> {
                performTask(taskId);
            });
        }
        
        executor.shutdown();
        try {
            executor.awaitTermination(5, java.util.concurrent.TimeUnit.SECONDS);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        }
        
        testSynchronization();
    }

    private int calculateFactorial(int n) {
        if (n <= 1) return 1;
        int result = 1;
        for (int i = 2; i <= n; i++) {
            result *= i;
        }
        return result;
    }

    private long calculateFibonacci(int n) {
        if (n <= 1) return n;
        long a = 0, b = 1;
        for (int i = 2; i <= n; i++) {
            long temp = a + b;
            a = b;
            b = temp;
        }
        return b;
    }

    private boolean checkPrime(int n) {
        if (n <= 1) return false;
        if (n <= 3) return true;
        if (n % 2 == 0 || n % 3 == 0) return false;
        
        for (int i = 5; i * i <= n; i += 6) {
            if (n % i == 0 || n % (i + 2) == 0) return false;
        }
        return true;
    }

    private double calculateAverage(double[] values) {
        double sum = 0;
        for (double value : values) {
            sum += value;
        }
        return sum / values.length;
    }

    private double findMaximum(double[] values) {
        double max = values[0];
        for (double value : values) {
            if (value > max) max = value;
        }
        return max;
    }

    private double findMinimum(double[] values) {
        double min = values[0];
        for (double value : values) {
            if (value < min) min = value;
        }
        return min;
    }

    private void performTask(int taskId) {
        try {
            Thread.sleep(100 + (int)(Math.random() * 200));
            System.out.println("Task " + taskId + " completed by " + Thread.currentThread().getName());
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        }
    }

    private void testSynchronization() {
        Object lock = new Object();
        
        Runnable task = () -> {
            synchronized (lock) {
                for (int i = 0; i < 5; i++) {
                    System.out.println(Thread.currentThread().getName() + " - " + i);
                    try {
                        Thread.sleep(50);
                    } catch (InterruptedException e) {
                        Thread.currentThread().interrupt();
                    }
                }
            }
        };
        
        Thread t1 = new Thread(task, "Thread-1");
        Thread t2 = new Thread(task, "Thread-2");
        
        t1.start();
        t2.start();
        
        try {
            t1.join();
            t2.join();
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        }
    }
}
