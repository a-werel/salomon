import io.shiftleft.codepropertygraph.generated.nodes

@main def exec(inputPath: String, outputPath: String) = {
    importCode(inputPath)

    def preorder(node: nodes.AstNode, variables: scala.collection.mutable.LinkedHashMap[String, String]): List[String] = {
        def variable(key: String): String = variables.getOrElseUpdate(key, s"VAR_${variables.size + 1}")
        val token = node match {
            case parameter: nodes.MethodParameterIn => s"METHOD_PARAMETER_IN:${variable(s"decl:${parameter.id}")}"
            case local: nodes.Local => s"LOCAL:${variable(s"decl:${local.id}")}"
            case identifier: nodes.Identifier =>
                val declarations = Iterator.single(identifier).refsTo.l
                val key = declarations.headOption match {
                    case Some(declaration) => s"decl:${declaration.id}"
                    case None => s"unresolved:${identifier.name}"
                }
                s"IDENTIFIER:${variable(key)}"
            case call: nodes.Call       => s"CALL:${call.name}"
            case literal: nodes.Literal => s"LITERAL:${literal.code}"
            case control: nodes.ControlStructure => s"CONTROL:${control.controlStructureType}"
            case other => other.label
        }
        val children = Iterator.single(node).astChildren.l.sortBy(_.order)
        List(token) ++ children.flatMap(child => preorder(child, variables)) ++ List("END")
    }

    val methods = cpg.method.filterNot(_.isExternal)
        .filterNot(_.name.startsWith("<")).l.sortBy(m => (m.filename, m.fullName))
    val records = methods.map { method =>
        ujson.Obj(
            "source" -> method.filename,
            "function" -> method.name,
            "representation" -> "joern-ast-preorder-vars-v2",
            "tokens" -> ujson.Arr.from(preorder(method, scala.collection.mutable.LinkedHashMap.empty[String, String]))
        )
    }
    val path = java.nio.file.Paths.get(outputPath)
    java.nio.file.Files.createDirectories(path.getParent)
    java.nio.file.Files.writeString(path, ujson.write(ujson.Arr.from(records), indent = 2))
}
