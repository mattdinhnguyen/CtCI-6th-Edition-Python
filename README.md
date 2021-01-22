# Python Solutions to *Cracking the Coding Interview, 6th Edition*

These are **Python** solutions for the book [Cracking the Coding Interview, 6th Edition](https://www.careercup.com/book) by *Gayle Laakmann McDowell*.

## How to use?

To run the programs, just use the `python chapter_X/filename.py` command.

The test cases are included in the solution files.

## Contributions

Contributions welcome!

In general solutions should fall into one of the following three categories:
 - *algorithm demonstration*.  This is the primary type of problem and solution that the text is concerned with. As
 such, solutions should not use standard library functions in cases that would make it unnecessary to implement the
 algorithm.  The goal of these solutions should be to have an easy to understand solution that demonstrates
 understanding of the algorithm.
 - *python demonstration*.  We also accept solutions that solve the problem in a more practical way, using whatever
 standard library functions are available.  Please do not use any third party dependencies.  These solutions should
 also be easy to understand and good examples of pythonic ways of doing things.
 - *speed demonstration*.  These alternative solutions may be accepted if the fastest way to do something is not very
 readable or intuitive and thus it doesn't fit into the first two categories.


Ideally solutions would work with Python 3.6 or greater, be well formatted, and include tests.  But if not we'll still
be happy to work with you.  Our codebase is formatted with the [`black`](https://black.readthedocs.io/en/stable/) code
 formatter.

# CtCI-6th-Edition
Solutions for [Cracking the Coding Interview 6th Edition](http://www.amazon.com/Cracking-Coding-Interview-6th-Edition/dp/0984782850) by [Gayle Laakmann McDowell](http://www.gayle.com/).

Crowdsourcing solutions for every widely used programming language. **Contributions welcome**.

## Cloning

Solutions in Java are contained directly in this repo and are the same solutions found
in [the book](http://www.amazon.com/Cracking-Coding-Interview-6th-Edition/dp/0984782850). Solutions in other programming languages are contributed by the community and each have
their own dedicated repos which are referenced from this repo as git submodules. What this means for cloning:

- If you want to make a local clone of solutions in all languages, you should use the `--recursive` option:

        git clone --recursive https://github.com/careercup/CtCI-6th-Edition.git

- If you're only interested in the Java solutions:

        git clone https://github.com/careercup/CtCI-6th-Edition.git

- If you originally cloned without `--recursive`, and then later decide you want the git submodules too, run:

        git submodule update --init --recursive

## Contributing

### Work Flow
1. Fork the appropriate repo for your language to your GitHub user. (see [Where to submit pull requests](#where-to-submit-pull-requests))
2. Write quality code and lint if applicable.
3. Add tests if applicable.
4. Open a pull request and provide a descriptive comment for what you did.

### Where to submit pull requests

Pull requests pertaining to Java solutions should be submitted to the main [CtCI-6th-Edition repo](https://github.com/careercup/CtCI-6th-Edition). Please submit pull requests for all other languages to the appropriate language-specific repo.

- [CtCI-6th-Edition-Clojure](https://github.com/careercup/CtCI-6th-Edition-Clojure)
- [CtCI-6th-Edition-C](https://github.com/careercup/CtCI-6th-Edition-C)
- [CtCI-6th-Edition-cpp](https://github.com/careercup/CtCI-6th-Edition-cpp)
- [CtCI-6th-Edition-CSharp](https://github.com/careercup/CtCI-6th-Edition-CSharp)
- [CtCI-6th-Edition-Go](https://github.com/careercup/CtCI-6th-Edition-Go)
- [CtCI-6th-Edition-Groovy](https://github.com/careercup/CtCI-6th-Edition-Groovy)
- [CtCI-6th-Edition-Haskell](https://github.com/careercup/CtCI-6th-Edition-Haskell)
- [CtCI-6th-Edition-JavaScript](https://github.com/careercup/CtCI-6th-Edition-JavaScript)
- [CtCI-6th-Edition-JavaScript-ES2015](https://github.com/careercup/CtCI-6th-Edition-JavaScript-ES2015)
- [CtCI-6th-Edition-Julia](https://github.com/careercup/CtCI-6th-Edition-Julia)
- [CtCI-6th-Edition-Kotlin](https://github.com/careercup/CtCI-6th-Edition-Kotlin)
- [CtCI-6th-Edition-Objective-C](https://github.com/careercup/CtCI-6th-Edition-Objective-C)
- [CtCI-6th-Edition-php](https://github.com/careercup/CtCI-6th-Edition-php)
- [CtCI-6th-Edition-Python](https://github.com/careercup/CtCI-6th-Edition-Python)
- [CtCI-6th-Edition-Ruby](https://github.com/careercup/CtCI-6th-Edition-Ruby)
- [CtCI-6th-Edition-Swift](https://github.com/careercup/CtCI-6th-Edition-Swift)

### Adding a new Language

Solutions in other languages are welcome too and should follow this workflow:

1. Create the new repo under your own GitHub user account and start contributing solutions. The repo name should follow this naming convention: `CtCI-6th-Edition-<language>`.
2. Open an [issue on the CtCI-6th-Edition repo](https://github.com/careercup/CtCI-6th-Edition/issues) to request that your solution repo be promoted to join the careercup GitHub organization and referenced from the main repo as a git submodule.
3. If your request is approved, navigate to your repo's settings page and select the "Transfer Ownership" option, and specify "careercup" as the new owner.
