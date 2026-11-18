import io.shiftleft.semanticcpg.language._

@main def exec(cpgFile: String, outputPath: String, exportPdg: Boolean = false) = {
    importCpg(cpgFile)
    java.nio.file.Files.createDirectories(java.nio.file.Paths.get(outputPath))
    val methods = cpg.method.filterNot(_.isExternal)
        .filter(m => !m.name.startsWith("<") && m.lineNumber.isDefined && m.block.nonEmpty)
        .l.sortBy(m => (m.lineNumber.map(_.toInt).getOrElse(0), m.columnNumber.map(_.toInt).getOrElse(0), m.id))
    val records = methods.map { method =>
        val dot = s"$outputPath/${method.id}.dot"
        val startLine: Int = method.lineNumber.map(_.toInt).getOrElse(0)
        val endLine: Int = method.lineNumberEnd.map(_.toInt).getOrElse(0)
        val column: Int = method.columnNumber.map(_.toInt).getOrElse(0)
        if (exportPdg) {
            val graph = cpg.method.id(method.id).dotPdg.l.mkString("\n")
            java.nio.file.Files.write(
                java.nio.file.Paths.get(dot),
                graph.getBytes(java.nio.charset.StandardCharsets.UTF_8)
            )
        }
        ujson.Obj(
            "id" -> method.id.toString,
            "name" -> method.name,
            "full_name" -> method.fullName,
            "line" -> ujson.Num(startLine.toDouble),
            "end_line" -> ujson.Num(endLine.toDouble),
            "column" -> ujson.Num(column.toDouble),
            "pdg" -> dot
        )
    }
    java.nio.file.Files.write(
        java.nio.file.Paths.get(outputPath, "functions.json"),
        ujson.write(ujson.Arr.from(records)).getBytes(java.nio.charset.StandardCharsets.UTF_8)
    )
}
